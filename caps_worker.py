#!/usr/bin/env python
"""Capabilities eval worker: start vLLM server, run inspect evals in parallel."""

import asyncio
import json
import os
import signal
import sys
import zipfile
from pathlib import Path

import yaml


async def wait_for_server(base_url, server_proc, timeout=300):
    """Poll vLLM health endpoint until ready. Fail fast if server dies."""
    import aiohttp

    url = f"{base_url}/health"
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if server_proc.returncode is not None:
            raise RuntimeError(f"vLLM server exited with code {server_proc.returncode}")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        return True
        except (aiohttp.ClientError, asyncio.TimeoutError):
            pass
        await asyncio.sleep(2)
    raise TimeoutError(f"vLLM server not ready after {timeout}s")


def find_successful_eval(log_dir, task_name):
    """Check if a successful .eval file already exists for this task.

    Returns the path if found, None otherwise.
    """
    # task_name is like "inspect_evals/ifeval" — the filename contains the short name
    short_task = task_name.split("/")[-1]  # "ifeval", "math", "mbpp"
    for eval_file in log_dir.glob("*.eval"):
        if short_task not in eval_file.name:
            continue
        try:
            with zipfile.ZipFile(eval_file) as z:
                with z.open("header.json") as f:
                    header = json.load(f)
                if header.get("status") == "success":
                    return eval_file
        except (zipfile.BadZipFile, KeyError, json.JSONDecodeError):
            continue
    return None


async def run_eval(task, model_name, log_dir, max_connections, extra_args):
    """Run a single inspect eval as a subprocess."""
    cmd = [
        sys.executable, "-m", "inspect_ai", "eval",
        task,
        "--model", f"vllm/{model_name}",
        "--max-connections", str(max_connections),
        "--log-dir", str(log_dir),
        "--display", "none",
        *extra_args,
    ]
    print(f"  Starting: {task}")
    proc = await asyncio.create_subprocess_exec(*cmd)
    await proc.wait()
    status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
    print(f"  Done: {task} — {status}")
    return proc.returncode


async def main():
    model = json.loads(sys.argv[1])
    config_path = sys.argv[2]
    log_dir = Path(sys.argv[3])

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_id = model["id"]
    short_name = model["short_name"]
    revision = model.get("revision")
    vllm_cfg = config.get("vllm", {})
    tp = vllm_cfg.get("tensor_parallel_size", 1)
    max_connections = vllm_cfg.get("max_connections", 10)

    prefix = f"[{short_name}]"
    model_log_dir = log_dir / "logs" / short_name
    model_log_dir.mkdir(parents=True, exist_ok=True)

    # --- Check which evals need running ---
    evals_to_run = []
    for eval_cfg in config["evals"]:
        task = eval_cfg["task"]
        existing = find_successful_eval(model_log_dir, task)
        if existing:
            print(f"{prefix} Skipping {task}: already succeeded ({existing.name})")
            continue
        evals_to_run.append(eval_cfg)

    if not evals_to_run:
        print(f"{prefix} All evals already succeeded")
        (model_log_dir / ".done").touch()
        return

    # --- Start vLLM server ---
    # Use SLURM_JOB_ID to avoid port collisions when multiple jobs share a node
    port = 10000 + (int(os.environ.get("SLURM_JOB_ID", 0)) % 50000)
    server_cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_id,
        "--tensor-parallel-size", str(tp),
        "--port", str(port),
        "--disable-log-requests",
    ]
    if revision:
        server_cmd.extend(["--revision", revision])

    print(f"{prefix} Starting vLLM server on port {port}...")
    server_proc = await asyncio.create_subprocess_exec(*server_cmd)

    base_url = f"http://localhost:{port}/v1"
    os.environ["VLLM_BASE_URL"] = base_url

    try:
        await wait_for_server(f"http://localhost:{port}", server_proc)
        print(f"{prefix} vLLM server ready")

        # --- Run pending evals in parallel ---
        tasks = []
        for eval_cfg in evals_to_run:
            extra_args = []
            if "limit" in eval_cfg:
                extra_args.extend(["--limit", str(eval_cfg["limit"])])
            if "sandbox" in eval_cfg:
                extra_args.extend(["--sandbox", eval_cfg["sandbox"]])
            if "epochs" in eval_cfg:
                extra_args.extend(["--epochs", str(eval_cfg["epochs"])])
            for key, val in eval_cfg.get("task_args", {}).items():
                extra_args.extend(["-T", f"{key}={val}"])

            tasks.append(run_eval(
                eval_cfg["task"], model_id, model_log_dir,
                max_connections, extra_args,
            ))

        results = await asyncio.gather(*tasks)
        n_failed = sum(1 for r in results if r != 0)

        if n_failed:
            print(f"{prefix} {n_failed}/{len(results)} evals failed")
        else:
            # Check if ALL evals (including previously succeeded ones) are now done
            all_done = all(
                find_successful_eval(model_log_dir, e["task"]) is not None
                for e in config["evals"]
            )
            if all_done:
                print(f"{prefix} All evals completed successfully")
                (model_log_dir / ".done").touch()
            else:
                print(f"{prefix} Some evals still missing")

    finally:
        if server_proc.returncode is None:
            server_proc.send_signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(server_proc.wait(), timeout=15)
            except asyncio.TimeoutError:
                server_proc.kill()
        print(f"{prefix} vLLM server stopped")


if __name__ == "__main__":
    asyncio.run(main())
