#!/usr/bin/env python
"""Orchestrator for capabilities evals: submit per-model Slurm jobs."""

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

REPO_DIR = Path(__file__).parent.resolve()


def parse_args():
    parser = argparse.ArgumentParser(description="Run capabilities evals via Inspect AI")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--log-dir", type=str, default=None)
    return parser.parse_args()


def write_sbatch_script(model, config, config_path, log_dir, slurm_cfg):
    short_name = model["short_name"]
    vllm_cfg = config.get("vllm", {})
    tp = vllm_cfg.get("tensor_parallel_size", 1)
    job_name = slurm_cfg.get("job_name", "caps")
    partition = slurm_cfg.get("partition", "compute")
    timeout = slurm_cfg.get("timeout", "4:00:00")
    gpus_per_node = slurm_cfg.get("gpus_per_node", 8)
    cpus_per_node = slurm_cfg.get("cpus_per_node", 64)
    cpus = cpus_per_node // gpus_per_node * tp

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
set -a; source .env 2>/dev/null || true; set +a

uv run python caps_worker.py \\
    '{model_json}' \\
    '{config_path}' \\
    '{log_dir}'
"""
    fd, path = tempfile.mkstemp(suffix=".sh", prefix=f"caps_{short_name}_", dir=slurm_log_dir)
    with os.fdopen(fd, "w") as f:
        f.write(script)
    return path


async def slurm_worker(slot_id, queue, config, config_path, log_dir, slurm_cfg):
    while True:
        try:
            model = queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        short_name = model["short_name"]

        # Skip if already complete
        done_marker = log_dir / "logs" / short_name / ".done"
        if done_marker.exists():
            print(f"[slot {slot_id}] Skipping {short_name}: already done")
            queue.task_done()
            continue

        script_path = write_sbatch_script(model, config, config_path, log_dir, slurm_cfg)

        proc = await asyncio.create_subprocess_exec(
            "sbatch", "--parsable", "--wait", script_path,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
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


def summarize(config, log_dir):
    """Print a summary of eval results from inspect logs."""
    print("\n" + "=" * 60)
    print("RESULTS: Capabilities Evals")
    print("=" * 60)

    logs_dir = log_dir / "logs"
    for model in config["models"]:
        short_name = model["short_name"]
        model_log_dir = logs_dir / short_name
        done = (model_log_dir / ".done").exists() if model_log_dir.exists() else False
        status = "DONE" if done else "INCOMPLETE"
        print(f"  {short_name}: {status}")

        if not model_log_dir.exists():
            continue

        for eval_file in sorted(model_log_dir.glob("*.eval")):
            print(f"    {eval_file.name}")


async def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    slurm_cfg = config.get("slurm", {})

    if args.log_dir:
        log_dir = Path(args.log_dir).resolve()
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = (Path("runs") / f"caps_{timestamp}").resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.config).resolve()
    shutil.copy2(config_path, log_dir / "config.yaml")

    max_concurrent = slurm_cfg.get("max_concurrent", 4)

    print(f"Log dir: {log_dir}")
    print(f"Models: {len(config['models'])}")
    print(f"Evals: {', '.join(e['task'] for e in config['evals'])}")
    print(f"Max concurrent Slurm jobs: {max_concurrent}")
    print()

    queue = asyncio.Queue()
    for model in config["models"]:
        queue.put_nowait(model)

    await asyncio.gather(
        *[
            slurm_worker(i, queue, config, config_path, log_dir, slurm_cfg)
            for i in range(max_concurrent)
        ]
    )

    summarize(config, log_dir)


if __name__ == "__main__":
    asyncio.run(main())
