#!/usr/bin/env python
"""Logit Diff Amplification (LDA) generation.

Amplifies behavioral differences between two model checkpoints:
    logits_amplified = logits_after + α(logits_after - logits_before)

Reference: https://www.goodfire.ai/research/model-diff-amplification
"""

import json
from collections import defaultdict

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, DynamicCache


def expand_kv_cache(past_key_values, n):
    """Expand a DynamicCache from batch=1 to batch=n."""
    new_cache = DynamicCache()
    for layer_idx in range(len(past_key_values)):
        key, value = past_key_values[layer_idx]
        new_cache.update(
            key.expand(n, -1, -1, -1).contiguous(),
            value.expand(n, -1, -1, -1).contiguous(),
            layer_idx,
        )
    return new_cache


def sample_top_p(logits, temperature, top_p):
    """Temperature + nucleus (top-p) sampling.

    Args:
        logits: [batch, vocab]
        temperature: float > 0
        top_p: float in (0, 1]

    Returns:
        [batch, 1] sampled token IDs
    """
    if temperature > 0:
        logits = logits / temperature

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_mask = (cumulative_probs - torch.softmax(sorted_logits, dim=-1)) >= top_p
    sorted_logits[sorted_mask] = float("-inf")

    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(1, sorted_indices, sorted_logits)

    probs = torch.softmax(filtered, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate_one_prompt(
    model_after, model_before, input_ids, alpha,
    n_samples, temperature, top_p, max_tokens,
    eos_token_id, pad_token_id,
    stream_a, stream_b,
):
    """Generate n_samples completions for one prompt using LDA.

    Args:
        input_ids: [1, seq_len] tokenized prompt
    Returns:
        list of n_samples token-id lists (variable length, EOS trimmed)
    """
    device = input_ids.device

    # Prefill: overlap both models on separate CUDA streams
    with torch.cuda.stream(stream_a):
        out_after = model_after(input_ids, use_cache=True)
    with torch.cuda.stream(stream_b):
        out_before = model_before(input_ids, use_cache=True)
    torch.cuda.synchronize()

    # Amplify first-token logits
    logits_a = out_after.logits[:, -1, :]
    logits_b = out_before.logits[:, -1, :]
    amplified = torch.clamp(logits_a + alpha * (logits_a - logits_b), -100.0, 100.0)

    amplified = amplified.expand(n_samples, -1)
    first_tokens = sample_top_p(amplified, temperature, top_p)  # [n, 1]

    kv_after = expand_kv_cache(out_after.past_key_values, n_samples)
    kv_before = expand_kv_cache(out_before.past_key_values, n_samples)

    # Decode — pre-allocate output tensor
    generated = torch.full((n_samples, max_tokens), pad_token_id, dtype=torch.long, device=device)
    generated[:, 0] = first_tokens.squeeze(-1)
    finished = torch.zeros(n_samples, dtype=torch.bool, device=device)
    next_input = first_tokens
    gen_len = 1

    for _ in range(max_tokens - 1):
        if finished.all():
            break

        with torch.cuda.stream(stream_a):
            out_after = model_after(next_input, past_key_values=kv_after, use_cache=True)
        with torch.cuda.stream(stream_b):
            out_before = model_before(next_input, past_key_values=kv_before, use_cache=True)
        torch.cuda.synchronize()

        kv_after = out_after.past_key_values
        kv_before = out_before.past_key_values

        logits_a = out_after.logits[:, -1, :]
        logits_b = out_before.logits[:, -1, :]
        amp = torch.clamp(logits_a + alpha * (logits_a - logits_b), -100.0, 100.0)

        amp[finished] = float("-inf")
        amp[finished, pad_token_id] = 0.0

        next_tokens = sample_top_p(amp, temperature, top_p)  # [n, 1]
        generated[:, gen_len] = next_tokens.squeeze(-1)
        gen_len += 1
        finished = finished | (next_tokens.squeeze(-1) == eos_token_id)
        next_input = next_tokens

    results = []
    for i in range(n_samples):
        tokens = generated[i, :gen_len].tolist()
        if eos_token_id in tokens:
            tokens = tokens[: tokens.index(eos_token_id)]
        results.append(tokens)
    return results


def load_completed(comp_path, n_samples):
    """Load completed completions from an incremental file.

    Reads the file once. Returns completed prompt texts (only prompts with
    exactly n_samples completions) and rewrites the file to discard any
    partial/orphaned rows so appends from a resumed run stay clean.
    """
    if not comp_path.exists():
        return {}

    texts_by_prompt = defaultdict(list)
    lines_by_prompt = defaultdict(list)
    with open(comp_path) as f:
        for line in f:
            try:
                row = json.loads(line)
                pid = row["prompt_id"]
                texts_by_prompt[pid].append(row["text"])
                lines_by_prompt[pid].append(line)
            except json.JSONDecodeError:
                break

    # Keep only fully-complete prompts
    complete = {
        pid: texts[:n_samples]
        for pid, texts in texts_by_prompt.items()
        if len(texts) >= n_samples
    }

    # Rewrite file without partial/orphaned rows
    if set(complete) != set(texts_by_prompt):
        with open(comp_path, "w") as f:
            for pid in complete:
                f.writelines(lines_by_prompt[pid][:n_samples])

    return complete


async def generate_lda(
    model_id, base_model_id, alpha,
    prompts, formatted, tokenizer,
    n_samples, temperature, top_p, max_tokens,
    revision=None, base_revision=None, prefix="",
    comp_path=None,
):
    """Generate completions using Logit Diff Amplification.

    Args:
        comp_path: If provided, save completions incrementally and resume
                   from partial results on resubmit.

    Returns:
        (completions, failed_prompts) where completions is
        {prompt_id: [text, ...]} and failed_prompts is [prompt_id, ...]
    """
    # Resume: single read, get completed texts and clean up partial rows
    completions = {}
    if comp_path:
        comp_path.parent.mkdir(parents=True, exist_ok=True)
        completions = load_completed(comp_path, n_samples)
        if completions:
            print(f"{prefix} Resuming — {len(completions)} prompts already on disk")

    done_ids = set(completions)
    remaining = [(p, f) for p, f in zip(prompts, formatted) if p["id"] not in done_ids]

    if not remaining:
        print(f"{prefix} All prompts already completed")
        return completions, []

    print(f"{prefix} Loading LDA models (α={alpha})...")
    model_after = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision,
        dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2",
    )
    model_before = AutoModelForCausalLM.from_pretrained(
        base_model_id, revision=base_revision,
        dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2",
    )
    model_after.eval()
    model_before.eval()

    print(f"{prefix} Compiling models with torch.compile...")
    model_after = torch.compile(model_after, mode="default")
    model_before = torch.compile(model_before, mode="default")

    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    failed_prompts = []
    gen_bar = tqdm(
        total=len(remaining) * n_samples,
        desc=f"{prefix} Generating (LDA α={alpha})",
        unit="comp",
    )

    for prompt, fmt_text in remaining:
        try:
            input_ids = tokenizer.encode(fmt_text, return_tensors="pt")
            input_ids = input_ids.to(model_after.device)

            token_lists = generate_one_prompt(
                model_after, model_before, input_ids, alpha,
                n_samples, temperature, top_p, max_tokens,
                eos_id, pad_id,
                stream_a, stream_b,
            )

            texts = [
                tokenizer.decode(toks, skip_special_tokens=True)
                for toks in token_lists
            ]
            completions[prompt["id"]] = texts
            gen_bar.update(n_samples)

            if comp_path:
                with open(comp_path, "a") as f:
                    for idx, text in enumerate(texts):
                        json.dump({"prompt_id": prompt["id"], "completion_idx": idx, "text": text}, f)
                        f.write("\n")

        except Exception as e:
            print(f"\n{prefix} Error prompt {prompt['id']}: {e}")
            failed_prompts.append(prompt["id"])
            gen_bar.update(n_samples)

    gen_bar.close()

    del model_after, model_before
    torch.cuda.empty_cache()

    return completions, failed_prompts
