#!/usr/bin/env python
"""Orchestrator: spawn independent GPU workers pulling from a shared model queue."""

import argparse
import asyncio
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="Run FORTRESS eval awareness experiment")
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Output directory for this run. Default: runs/<timestamp>",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file",
    )
    return parser.parse_args()


async def gpu_worker(gpu_ids, queue, config, config_path, prompts_path, log_dir):
    """Pull models from shared queue until empty. Fully independent of other workers."""
    while True:
        try:
            model = queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        score_path = log_dir / "scores" / f"{model['short_name']}.jsonl"
        if score_path.exists():
            n_lines = sum(1 for _ in open(score_path))
            expected = config["sampling"]["n"] * config["dataset"]["num_prompts"]
            if n_lines >= expected:
                print(f"[GPU {gpu_ids}] Skipping {model['short_name']}: already scored ({n_lines} lines)")
                queue.task_done()
                continue
            else:
                print(f"[GPU {gpu_ids}] Incomplete {model['short_name']}: {n_lines}/{expected} lines, re-running")
                score_path.unlink()
                comp_path = log_dir / "completions" / f"{model['short_name']}.jsonl"
                if comp_path.exists():
                    comp_path.unlink()

        print(f"[GPU {gpu_ids}] Starting {model['short_name']}...")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_ids}
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "worker.py",
            json.dumps(model),
            str(config_path),
            str(prompts_path),
            str(log_dir),
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        ret = await proc.wait()
        queue.task_done()
        if ret != 0:
            print(f"[GPU {gpu_ids}] FAILED: {model['short_name']} (exit {ret})")
        else:
            print(f"[GPU {gpu_ids}] Done: {model['short_name']}")


def analyze(config, log_dir):
    print("\n" + "=" * 60)
    print("RESULTS: Eval Awareness Rate")
    print("=" * 60)

    results = []
    score_dir = log_dir / "scores"

    for model in config["models"]:
        path = score_dir / f"{model['short_name']}.jsonl"
        if not path.exists():
            print(f"  {model['short_name']:>15}: NO DATA")
            continue

        scores = [json.loads(line) for line in open(path)]
        total = len(scores)
        aware = sum(1 for s in scores if s["aware"] is True)
        failed = sum(1 for s in scores if s["aware"] is None)
        valid = total - failed
        rate = aware / valid if valid > 0 else 0

        results.append(
            {
                "model": model["short_name"],
                "total": total,
                "aware": aware,
                "failed": failed,
                "awareness_rate": rate,
            }
        )
        print(
            f"  {model['short_name']:>15}: {rate:6.2%}  ({aware}/{valid}, {failed} failed)"
        )

    csv_path = log_dir / "summary.csv"
    with open(csv_path, "w") as f:
        f.write("model,total,aware,failed,awareness_rate\n")
        for r in results:
            f.write(
                f"{r['model']},{r['total']},{r['aware']},{r['failed']},{r['awareness_rate']:.4f}\n"
            )
    print(f"\nSaved -> {csv_path}")


async def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Resolve log dir
    if args.log_dir:
        log_dir = Path(args.log_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path("runs") / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot config into log dir for reproducibility
    shutil.copy2(args.config, log_dir / "config.yaml")

    prompts_path = Path(config["paths"]["prompts"])
    if not prompts_path.exists():
        print(f"ERROR: {prompts_path} not found. Run `uv run python sample_prompts.py` first.")
        sys.exit(1)

    # Copy prompts into log dir for self-contained runs
    shutil.copy2(prompts_path, log_dir / "prompts.jsonl")

    # Validate GPU config
    tp = config["vllm"]["tensor_parallel_size"]
    for gpu_ids in config["vllm"]["gpu_sets"]:
        n_gpus = len(gpu_ids.split(","))
        if n_gpus != tp:
            print(
                f"ERROR: gpu_set '{gpu_ids}' has {n_gpus} GPUs but "
                f"tensor_parallel_size is {tp}"
            )
            sys.exit(1)

    print(f"Log dir: {log_dir}")

    # Shared work queue — workers pull from it, whoever finishes first grabs the next
    queue = asyncio.Queue()
    for model in config["models"]:
        queue.put_nowait(model)

    gpu_sets = config["vllm"]["gpu_sets"]
    await asyncio.gather(
        *[
            gpu_worker(gpu_ids, queue, config, args.config, prompts_path, log_dir)
            for gpu_ids in gpu_sets
        ]
    )

    analyze(config, log_dir)


if __name__ == "__main__":
    asyncio.run(main())
