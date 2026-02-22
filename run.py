#!/usr/bin/env python
"""Orchestrator: spawn independent GPU workers pulling from a shared model queue."""

import asyncio
import json
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()


async def gpu_worker(gpu_ids, queue, config_path, prompts_path, scores_dir):
    """Pull models from shared queue until empty. Fully independent of other workers."""
    while True:
        try:
            model = queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        score_path = Path(scores_dir) / f"{model['short_name']}.jsonl"
        if score_path.exists():
            print(f"[GPU {gpu_ids}] Skipping {model['short_name']}: already scored")
            queue.task_done()
            continue

        print(f"[GPU {gpu_ids}] Starting {model['short_name']}...")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_ids}
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "worker.py",
            json.dumps(model),
            config_path,
            prompts_path,
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


def analyze(config):
    print("\n" + "=" * 60)
    print("RESULTS: Eval Awareness Rate")
    print("=" * 60)

    results = []
    score_dir = Path(config["paths"]["scores"])

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

    results_dir = Path(config["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "summary.csv"
    with open(csv_path, "w") as f:
        f.write("model,total,aware,failed,awareness_rate\n")
        for r in results:
            f.write(
                f"{r['model']},{r['total']},{r['aware']},{r['failed']},{r['awareness_rate']:.4f}\n"
            )
    print(f"\nSaved -> {csv_path}")


async def main():
    config_path = "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    prompts_path = config["paths"]["prompts"]
    if not Path(prompts_path).exists():
        print(f"ERROR: {prompts_path} not found. Run `uv run python sample_prompts.py` first.")
        sys.exit(1)

    # Shared work queue — workers pull from it, whoever finishes first grabs the next
    queue = asyncio.Queue()
    for model in config["models"]:
        queue.put_nowait(model)

    gpu_sets = config["vllm"]["gpu_sets"]
    await asyncio.gather(
        *[
            gpu_worker(gpu_ids, queue, config_path, prompts_path, config["paths"]["scores"])
            for gpu_ids in gpu_sets
        ]
    )

    analyze(config)


if __name__ == "__main__":
    asyncio.run(main())
