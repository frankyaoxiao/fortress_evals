#!/usr/bin/env python
"""Sample prompts from FORTRESS dataset. Run once before run.py."""

import argparse
import json
import random
from pathlib import Path

import pyarrow as pa
import yaml


def main():
    parser = argparse.ArgumentParser(description="Sample prompts from FORTRESS")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    path = Path(config["paths"]["prompts"])
    if path.exists():
        n = sum(1 for _ in open(path))
        print(f"Already exists: {path} ({n} prompts)")
        return

    arrow_path = config["dataset"]["arrow_path"]
    reader = pa.ipc.open_stream(arrow_path)
    table = reader.read_all()

    n_total = table.num_rows
    n_sample = config["dataset"]["num_prompts"]
    rng = random.Random(config["dataset"]["seed"])
    indices = rng.sample(range(n_total), n_sample)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i in indices:
            json.dump(
                {
                    "id": table.column("ID")[i].as_py(),
                    "adversarial_prompt": table.column("adversarial_prompt")[i].as_py(),
                    "benign_prompt": table.column("benign_prompt")[i].as_py(),
                    "risk_domain": table.column("risk_domain")[i].as_py(),
                    "risk_subdomain": table.column("risk_subdomain")[i].as_py(),
                },
                f,
            )
            f.write("\n")
    print(f"Sampled {n_sample}/{n_total} prompts -> {path}")


if __name__ == "__main__":
    main()
