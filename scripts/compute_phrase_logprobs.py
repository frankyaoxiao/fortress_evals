#!/usr/bin/env python3
"""Compute per-prompt logprobs of a phrase appended after the chat-template start-of-response.

For each prompt in a dataset, construct:
    chat_template(prompt, add_generation_prompt=True) + phrase_tokens

Then use vLLM's prompt_logprobs=1 to get the per-token conditional logprob of each
phrase token given the preceding context. Save per-token and sum logprobs to JSONL.
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


DEFAULT_DATASET = "/data/artifacts/frank/datasets/Dolci-Think-RL-7B-with-messages-hf-ifeval-only-hf/data/train-00000-of-00001.parquet"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF path or local checkpoint dir")
    p.add_argument("--phrase", default="The user might be testing",
                   help="Phrase to measure logprob for")
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--limit", type=int, default=None, help="Limit number of prompts (for testing)")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = {
        "model": args.model,
        "phrase": args.phrase,
        "dataset": args.dataset,
        "limit": args.limit,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Load tokenizer and dataset
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Encode phrase — include leading space so it follows naturally after <think>\n
    # But when appending tokens after a prompt that ends with "\n", we want the phrase to start with "The" not " The"
    # Let's encode without leading space but check both
    phrase_ids = tokenizer.encode(args.phrase, add_special_tokens=False)
    phrase_token_strs = [tokenizer.decode([t]) for t in phrase_ids]
    print(f"Phrase: '{args.phrase}'")
    print(f"Phrase tokens ({len(phrase_ids)}): {phrase_token_strs}")

    print(f"Loading dataset: {args.dataset}")
    df = pd.read_parquet(args.dataset)
    if args.limit:
        df = df.head(args.limit)
    print(f"Loaded {len(df)} prompts")

    # Build inputs
    print("Building prompt sequences...")
    inputs = []
    dataset_idxs = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        prompt_text = row["prompt"]
        # The dataset prompt starts with "user: " already; we want the user content
        if prompt_text.startswith("user: "):
            user_content = prompt_text[6:]
        else:
            user_content = prompt_text
        # Apply chat template, get tokens as a string, then tokenize
        prefix_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            add_generation_prompt=True,
            tokenize=False,
        )
        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        full_ids = list(prefix_ids) + list(phrase_ids)
        inputs.append({
            "prompt_token_ids": full_ids,
            "prefix_len": len(prefix_ids),
        })
        dataset_idxs.append(idx)

    # Load vLLM
    print(f"Loading vLLM model: {args.model}")
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
    )

    # Generate with prompt_logprobs=1 — returns logprob for each prompt token
    sampling = SamplingParams(
        max_tokens=1,
        prompt_logprobs=1,
        temperature=0.0,
    )

    print(f"Generating logprobs for {len(inputs)} prompts...")
    requests = [{"prompt_token_ids": inp["prompt_token_ids"]} for inp in inputs]
    outputs = llm.generate(requests, sampling)

    # Extract phrase logprobs
    print("Extracting phrase logprobs...")
    results = []
    for didx, inp, out in tqdm(zip(dataset_idxs, inputs, outputs), total=len(inputs)):
        prefix_len = inp["prefix_len"]
        pl = out.prompt_logprobs  # list of dicts, one per prompt token position (None at position 0)
        token_lps = []
        for i, tid in enumerate(phrase_ids):
            pos = prefix_len + i
            if pos < len(pl) and pl[pos] is not None and tid in pl[pos]:
                token_lps.append(pl[pos][tid].logprob)
            else:
                token_lps.append(None)
        sum_lp = sum(lp for lp in token_lps if lp is not None)
        results.append({
            "dataset_idx": int(didx),
            "phrase_tokens": phrase_token_strs,
            "token_logprobs": token_lps,
            "sum_logprob": sum_lp,
        })

    # Save
    output_path = output_dir / "scores.jsonl"
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(results)} results to {output_path}")

    # Quick stats
    import numpy as np
    sums = np.array([r["sum_logprob"] for r in results])
    print(f"\nSum logprob stats: mean={sums.mean():.3f}, "
          f"median={np.median(sums):.3f}, std={sums.std():.3f}")
    print(f"Percentiles: 5%={np.percentile(sums, 5):.2f}, "
          f"25%={np.percentile(sums, 25):.2f}, "
          f"75%={np.percentile(sums, 75):.2f}, "
          f"95%={np.percentile(sums, 95):.2f}")


if __name__ == "__main__":
    main()
