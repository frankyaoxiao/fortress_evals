#!/usr/bin/env python
"""Orchestrator: submit per-model Slurm jobs from a shared queue."""

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

REPO_DIR = Path(__file__).parent.resolve()


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


def write_sbatch_script(model, config, config_path, prompts_path, log_dir, slurm_cfg):
    """Write a temporary sbatch script for one model. Returns the script path."""
    short_name = model["short_name"]
    tp = config["vllm"]["tensor_parallel_size"]
    job_name = slurm_cfg.get("job_name", "fortress")
    partition = slurm_cfg.get("partition", "compute")
    timeout = slurm_cfg.get("timeout", "4:00:00")
    gpus_per_node = slurm_cfg.get("gpus_per_node", 8)
    cpus_per_node = slurm_cfg.get("cpus_per_node", 64)
    cpus = cpus_per_node // gpus_per_node * tp
    scorer_limit = config["scoring"]["max_concurrent"] // slurm_cfg.get("max_concurrent", 4)

    slurm_log_dir = log_dir / "slurm_logs"
    slurm_log_dir.mkdir(parents=True, exist_ok=True)

    model_json = json.dumps(model)

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}_{short_name}
#SBATCH --nodes=1
#SBATCH --gpus-per-node={tp}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --partition={partition}
#SBATCH --time={timeout}
#SBATCH --output={slurm_log_dir}/{short_name}.out
#SBATCH --error={slurm_log_dir}/{short_name}.err

set -euo pipefail
cd {REPO_DIR}

export SCORER_MAX_CONCURRENT={scorer_limit}

uv run python worker.py \\
    '{model_json}' \\
    '{config_path}' \\
    '{prompts_path}' \\
    '{log_dir}'
"""
    fd, path = tempfile.mkstemp(suffix=".sh", prefix=f"fortress_{short_name}_", dir=slurm_log_dir)
    with os.fdopen(fd, "w") as f:
        f.write(script)
    return path


async def slurm_worker(slot_id, queue, config, config_path, prompts_path, log_dir, slurm_cfg, n_prompts):
    """Pull models from shared queue, submit sbatch --wait for each."""
    while True:
        try:
            model = queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        short_name = model["short_name"]

        # Per-model prompts override
        if "prompts" in model:
            model_prompts = Path(model["prompts"]).resolve()
            model_n_prompts = sum(1 for _ in open(model_prompts))
        else:
            model_prompts = prompts_path
            model_n_prompts = n_prompts

        # Skip if already complete
        score_path = log_dir / "scores" / f"{short_name}.jsonl"
        if score_path.exists():
            n_lines = sum(1 for _ in open(score_path))
            expected = config["sampling"]["n"] * model_n_prompts
            if n_lines >= expected:
                print(f"[slot {slot_id}] Skipping {short_name}: already scored ({n_lines} lines)")
                queue.task_done()
                continue
            else:
                print(f"[slot {slot_id}] Incomplete {short_name}: {n_lines}/{expected} lines, re-running")
                score_path.unlink()

        script_path = write_sbatch_script(
            model, config, config_path, model_prompts, log_dir, slurm_cfg
        )

        # sbatch --parsable --wait: prints job ID immediately, blocks until done
        proc = await asyncio.create_subprocess_exec(
            "sbatch", "--parsable", "--wait", script_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        job_id_line = await proc.stdout.readline()
        job_id = job_id_line.decode().strip()
        print(f"[slot {slot_id}] Submitted {short_name} (job {job_id})")

        ret = await proc.wait()
        queue.task_done()

        if ret == 0:
            print(f"[slot {slot_id}] Done: {short_name} (job {job_id})")
        else:
            print(f"[slot {slot_id}] FAILED: {short_name} (job {job_id}, exit {ret})")


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

    slurm_cfg = config.get("slurm", {})

    # Resolve all paths to absolute — sbatch scripts run on different nodes
    if args.log_dir:
        log_dir = Path(args.log_dir).resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = (Path("runs") / timestamp).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.config).resolve()
    shutil.copy2(config_path, log_dir / "config.yaml")

    prompts_path = Path(config["paths"]["prompts"]).resolve()
    if not prompts_path.exists():
        print(f"ERROR: {prompts_path} not found.")
        sys.exit(1)

    shutil.copy2(prompts_path, log_dir / "prompts.jsonl")

    n_prompts = sum(1 for _ in open(prompts_path))
    max_concurrent = slurm_cfg.get("max_concurrent", 4)

    print(f"Log dir: {log_dir}")
    print(f"Prompts: {n_prompts} (from {prompts_path})")
    print(f"Models: {len(config['models'])}")
    print(f"Max concurrent Slurm jobs: {max_concurrent}")
    print()

    # Shared work queue
    queue = asyncio.Queue()
    for model in config["models"]:
        queue.put_nowait(model)

    await asyncio.gather(
        *[
            slurm_worker(i, queue, config, config_path, prompts_path, log_dir, slurm_cfg, n_prompts)
            for i in range(max_concurrent)
        ]
    )

    analyze(config, log_dir)


if __name__ == "__main__":
    asyncio.run(main())
