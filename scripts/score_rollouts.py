#!/usr/bin/env python3
"""Score training rollouts for eval awareness.

Loads rollouts from a training run, optionally labels them as persona-filtered
or kept, scores with GPT-5-mini, and saves results.

Usage:
    python scripts/score_rollouts.py \
        --run olmo3-7b-think-rl-ifeval-persona-from200 \
        --rollout-steps 399 424 \
        --filter-steps 400 425 \
        --output runs/rollout_scoring/pf200_transition.jsonl

The --rollout-steps and --filter-steps must correspond (offset by 1).
If --filter-steps is omitted, no filtered/kept labeling is done.
"""

import argparse
import asyncio
import json
import glob
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoTokenizer

# Reuse scorer from worker.py
sys.path.insert(0, str(Path(__file__).parent.parent))
from worker import SCORER_PROMPT, parse_scorer_response, make_scorer, score_one

load_dotenv()

BASE_DIR = "/data/artifacts/frank/openinstruct"
DATASET_PATH = "/data/artifacts/frank/datasets/Dolci-Think-RL-7B-with-messages-hf-ifeval-only-hf/data/train-00000-of-00001.parquet"


def parse_args():
    parser = argparse.ArgumentParser(description="Score training rollouts for eval awareness")
    parser.add_argument("--run", required=True, help="Run directory name under BASE_DIR")
    parser.add_argument("--rollout-steps", type=int, nargs="+", required=True,
                        help="Rollout step range (start end), inclusive")
    parser.add_argument("--filter-steps", type=int, nargs="+", default=None,
                        help="Corresponding filter training_step range (start end), inclusive. Offset +1 from rollout steps.")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--scorer-model", default="gpt-5-mini")
    parser.add_argument("--max-concurrent", type=int, default=400)
    parser.add_argument("--tokenizer", default="allenai/Olmo-3-7B-Think-DPO")
    return parser.parse_args()


def load_rollouts(run_dir, step_start, step_end):
    """Load all rollouts in the step range [step_start, step_end]."""
    rollout_dir = os.path.join(run_dir, "rollouts")
    rollout_files = sorted(glob.glob(os.path.join(rollout_dir, "*_rollouts_*.jsonl")))

    rollouts = []
    for f in rollout_files:
        with open(f) as fh:
            for line in fh:
                d = json.loads(line)
                if step_start <= d["step"] <= step_end:
                    rollouts.append(d)
    return rollouts


def load_filtered_set(run_dir, step_start, step_end):
    """Load filtered (training_step, dataset_idx) pairs in range."""
    # Find the persona filtered rollouts directory
    subdirs = [d for d in os.listdir(run_dir)
               if os.path.isdir(os.path.join(run_dir, d)) and "persona_filtered" not in d
               and d not in ("checkpoint_states", "rollouts")]
    if not subdirs:
        return set()

    filtered_dir = os.path.join(run_dir, subdirs[0])
    filtered_files = glob.glob(os.path.join(filtered_dir, "persona_filtered_rollouts_rank*.jsonl"))

    filtered = set()
    for f in filtered_files:
        with open(f) as fh:
            for line in fh:
                d = json.loads(line)
                ts = d["training_step"]
                if step_start <= ts <= step_end:
                    for item in d["filtered"]:
                        filtered.add((ts, item["dataset_idx"]))
    return filtered


def build_prompt_to_dataset_idx(dataset_path):
    """Build a mapping from user message text (first 200 chars) to dataset index."""
    df = pd.read_parquet(dataset_path)
    mapping = {}
    for idx, row in df.iterrows():
        prompt = row["prompt"]
        user_msg = prompt[6:] if prompt.startswith("user: ") else prompt
        mapping[user_msg[:200]] = idx
    return mapping


def decode_and_match(rollouts, tokenizer, prompt_to_didx):
    """Decode prompt_tokens, extract user message, match to dataset_idx."""
    for r in rollouts:
        decoded = tokenizer.decode(r["prompt_tokens"], skip_special_tokens=False)
        parts = decoded.split("<|im_start|>user")
        if len(parts) > 1:
            user_msg = parts[-1].split("<|im_end|>")[0].strip()
            key = user_msg[:200]
            r["dataset_idx"] = prompt_to_didx.get(key)
            r["prompt_text"] = user_msg
        else:
            r["dataset_idx"] = None
            r["prompt_text"] = decoded

        # Decode response
        r["response_text"] = tokenizer.decode(r["response_tokens"], skip_special_tokens=True)

    return rollouts


async def score_all(rollouts, scoring_config, max_concurrent):
    """Score all rollouts for eval awareness."""
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    sem = asyncio.Semaphore(max_concurrent)
    bar = tqdm(total=len(rollouts), desc="Scoring")

    async def score_rollout(r):
        result = await score_one(client, sem, r["response_text"], scoring_config, bar)
        if result:
            r["aware"] = result["aware"]
            r["reasoning"] = result["reasoning"]
            r["quote"] = result["quote"]
        else:
            r["aware"] = None
            r["reasoning"] = None
            r["quote"] = None
        return r

    tasks = [score_rollout(r) for r in rollouts]
    return await asyncio.gather(*tasks)


def main():
    args = parse_args()
    run_dir = os.path.join(BASE_DIR, args.run)

    step_start = args.rollout_steps[0]
    step_end = args.rollout_steps[1] if len(args.rollout_steps) > 1 else args.rollout_steps[0]

    # Save config — output is a directory, scores go inside it
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "scores.jsonl"
    config = {
        "run": args.run,
        "run_dir": run_dir,
        "rollout_steps": [step_start, step_end],
        "filter_steps": args.filter_steps,
        "scorer_model": args.scorer_model,
        "max_concurrent": args.max_concurrent,
        "tokenizer": args.tokenizer,
        "dataset_path": DATASET_PATH,
        "output": str(output_dir),
    }
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")

    print(f"Loading tokenizer {args.tokenizer}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f"Loading rollouts for steps {step_start}-{step_end}...")
    rollouts = load_rollouts(run_dir, step_start, step_end)
    print(f"  Loaded {len(rollouts)} rollouts")

    # Load filtered set if requested
    filtered_set = set()
    if args.filter_steps:
        fs_start = args.filter_steps[0]
        fs_end = args.filter_steps[1] if len(args.filter_steps) > 1 else args.filter_steps[0]
        print(f"Loading filtered set for training_steps {fs_start}-{fs_end}...")
        filtered_set = load_filtered_set(run_dir, fs_start, fs_end)
        print(f"  Loaded {len(filtered_set)} filtered (step, dataset_idx) pairs")

    print("Building prompt-to-dataset-idx mapping...")
    prompt_to_didx = build_prompt_to_dataset_idx(DATASET_PATH)

    print("Decoding and matching rollouts...")
    rollouts = decode_and_match(rollouts, tokenizer, prompt_to_didx)

    matched = sum(1 for r in rollouts if r["dataset_idx"] is not None)
    print(f"  Matched {matched}/{len(rollouts)} to dataset indices")

    # Label filtered/kept
    if filtered_set:
        for r in rollouts:
            if r["dataset_idx"] is not None:
                # filter key: (training_step = rollout_step + 1, dataset_idx)
                r["persona_filtered"] = (r["step"] + 1, r["dataset_idx"]) in filtered_set
            else:
                r["persona_filtered"] = None

        n_filtered = sum(1 for r in rollouts if r.get("persona_filtered") is True)
        n_kept = sum(1 for r in rollouts if r.get("persona_filtered") is False)
        print(f"  Filtered: {n_filtered}, Kept: {n_kept}, Unknown: {len(rollouts) - n_filtered - n_kept}")

    # Score
    scoring_config = {
        "model": args.scorer_model,
        "max_completion_tokens": 2048,
        "reasoning_effort": "medium",
    }
    print(f"Scoring {len(rollouts)} rollouts with {args.scorer_model}...")
    scored = asyncio.run(score_all(rollouts, scoring_config, args.max_concurrent))

    # Save
    with open(output_path, "w") as f:
        for r in scored:
            row = {
                "step": r["step"],
                "sample_idx": r["sample_idx"],
                "prompt_idx": r["prompt_idx"],
                "dataset_idx": r["dataset_idx"],
                "reward": r["reward"],
                "advantage": r["advantage"],
                "prompt_text": r["prompt_text"],
                "response_text": r["response_text"],
                "aware": r["aware"],
                "reasoning": r["reasoning"],
                "quote": r["quote"],
            }
            if "persona_filtered" in r:
                row["persona_filtered"] = r["persona_filtered"]
            json.dump(row, f)
            f.write("\n")

    print(f"\nSaved {len(scored)} scored rollouts to {output_path}")

    # Summary
    aware_count = sum(1 for r in scored if r["aware"] is True)
    total_scored = sum(1 for r in scored if r["aware"] is not None)
    print(f"Overall: {aware_count}/{total_scored} = {100*aware_count/total_scored:.1f}% eval-aware")

    if filtered_set:
        for label, val in [("Filtered", True), ("Kept", False)]:
            subset = [r for r in scored if r.get("persona_filtered") == val]
            a = sum(1 for r in subset if r["aware"] is True)
            t = sum(1 for r in subset if r["aware"] is not None)
            if t > 0:
                print(f"  {label}: {a}/{t} = {100*a/t:.1f}% eval-aware")


if __name__ == "__main__":
    main()
