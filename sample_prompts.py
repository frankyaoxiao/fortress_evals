#!/usr/bin/env python
"""Sample prompts from FORTRESS dataset. Run once before run.py."""

import argparse
import json
from pathlib import Path

import yaml
from datasets import load_dataset


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

    ds = load_dataset(config["dataset"]["name"], split="train")
    ds = ds.shuffle(seed=config["dataset"]["seed"]).select(
        range(config["dataset"]["num_prompts"])
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in ds:
            json.dump(
                {
                    "id": row["ID"],
                    "adversarial_prompt": row["adversarial_prompt"],
                    "benign_prompt": row["benign_prompt"],
                    "risk_domain": row["risk_domain"],
                    "risk_subdomain": row["risk_subdomain"],
                },
                f,
            )
            f.write("\n")
    print(f"Sampled {len(ds)} prompts -> {path}")


if __name__ == "__main__":
    main()
