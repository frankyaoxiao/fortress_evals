#!/usr/bin/env python3
"""Build a filtered IFEval dataset by removing prompts with high logprob of ' testing'.

Reads per-prompt DPO logprobs from runs/phrase_logprobs/dpo/scores.jsonl, filters
the source parquet to rows whose ' testing' token logprob is below the threshold
(default: log(0.1) ≈ -2.303, i.e. max 10% probability), and writes the result as
a Hub-style parquet dataset matching the layout of the source.
"""

import argparse
import json
import math
import shutil
from pathlib import Path

import pyarrow.parquet as pq


SOURCE_DATASET = "/data/artifacts/frank/datasets/Dolci-Think-RL-7B-with-messages-hf-ifeval-only-hf"
SCORES_JSONL = "/home/fxiao/eval_awareness/fortress/runs/phrase_logprobs/dpo/scores.jsonl"
TESTING_IDX = 4  # position of ' testing' in phrase_tokens ["The", " user", " might", " be", " testing"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=SOURCE_DATASET,
                   help="Source dataset directory (Hub-style, contains data/*.parquet)")
    p.add_argument("--scores", default=SCORES_JSONL,
                   help="JSONL with per-prompt token_logprobs from compute_phrase_logprobs.py")
    p.add_argument("--output", required=True,
                   help="Output dataset directory; will contain data/train-00000-of-00001.parquet")
    p.add_argument("--max-prob", type=float, default=0.1,
                   help="Max probability of ' testing' token to keep a prompt (default 0.1)")
    return p.parse_args()


def main():
    args = parse_args()
    threshold_logprob = math.log(args.max_prob)

    # 1. Read scores and build set of dataset_idx to keep
    keep_idx = []
    drop_idx = []
    with open(args.scores) as f:
        for line in f:
            r = json.loads(line)
            lp = r["token_logprobs"][TESTING_IDX]
            if lp is None or lp <= threshold_logprob:
                keep_idx.append(r["dataset_idx"])
            else:
                drop_idx.append(r["dataset_idx"])
    keep_idx.sort()
    print(f"Threshold: ' testing' logprob <= log({args.max_prob}) = {threshold_logprob:.4f}")
    print(f"Kept: {len(keep_idx)}  Dropped: {len(drop_idx)}  Total: {len(keep_idx) + len(drop_idx)}")

    # 2. Read source parquet as pyarrow Table (preserves schema exactly)
    src_parquet = Path(args.source) / "data" / "train-00000-of-00001.parquet"
    print(f"\nReading source parquet: {src_parquet}")
    table = pq.read_table(src_parquet)
    print(f"Source rows: {table.num_rows}, columns: {len(table.column_names)}")
    assert table.num_rows == len(keep_idx) + len(drop_idx), (
        f"Row count mismatch: parquet has {table.num_rows} rows but scores has "
        f"{len(keep_idx) + len(drop_idx)}. dataset_idx <-> row position assumption is wrong."
    )

    # 3. Filter by row index (dataset_idx in scores.jsonl is the positional row index)
    filtered = table.take(keep_idx)
    print(f"Filtered rows: {filtered.num_rows}")
    assert filtered.num_rows == len(keep_idx)

    # 4. Write to output dir matching source layout (data/train-00000-of-00001.parquet only)
    out_dir = Path(args.output)
    out_data_dir = out_dir / "data"
    out_data_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = out_data_dir / "train-00000-of-00001.parquet"
    pq.write_table(filtered, out_parquet)
    print(f"\nWrote: {out_parquet}")
    print(f"Size: {out_parquet.stat().st_size / 1e6:.1f} MB")

    # 5. Save filter config alongside (for provenance, not loaded by training)
    config = {
        "source_dataset": str(args.source),
        "scores_jsonl": str(args.scores),
        "max_prob_testing": args.max_prob,
        "threshold_logprob": threshold_logprob,
        "phrase_token_idx": TESTING_IDX,
        "phrase_token": " testing",
        "source_rows": table.num_rows,
        "kept_rows": len(keep_idx),
        "dropped_rows": len(drop_idx),
    }
    with open(out_dir / "filter_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Wrote: {out_dir / 'filter_config.json'}")


if __name__ == "__main__":
    main()
