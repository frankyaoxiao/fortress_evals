#!/usr/bin/env python3
"""
Comprehensive analysis of degenerate repetition tails across LDA configurations.
"""
import json
import os
import re
import sys
import random
import time
from collections import defaultdict, Counter
from pathlib import Path

random.seed(42)

BASE_DIR = Path("/home/fxiao/eval_awareness/fortress/runs")

# ── Configuration ──────────────────────────────────────────────────────
CONFIGS = {
    "baseline (no LDA)": {
        "dir": "hbsr_7b_steps/completions",
        "alpha": 0.0,
    },
    "a=0.4": {
        "dir": "lda_hbsr_7b_steps/completions",
        "alpha": 0.4,
    },
    "a=0.8": {
        "dir": "lda08_hbsr_7b_steps/completions",
        "alpha": 0.8,
    },
    "a=1.2 HF": {
        "dir": "lda12_hbsr_7b_steps/completions",
        "alpha": 1.2,
    },
    "a=1.2 Local": {
        "dir": "lda12_hbsr_7b_local_steps/completions",
        "alpha": 1.2,
    },
}

# ── Degenerate detection functions ────────────────────────────────────

def detect_low_unique_word_ratio(text, tail_len=200, threshold=0.3):
    """Method A: Low unique word ratio in last 200 chars."""
    tail = text[-tail_len:]
    words = tail.split()
    if len(words) < 5:
        return False
    unique_ratio = len(set(w.lower() for w in words)) / len(words)
    return unique_ratio < threshold


def detect_phrase_repetition(text, tail_len=500, min_reps=10):
    """Method B: Last 500 chars contain 10+ repetitions of any 2-3 word phrase."""
    tail = text[-tail_len:].lower()
    words = tail.split()
    for n in (2, 3):
        if len(words) < n:
            continue
        phrase_counts = Counter()
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i:i+n])
            phrase_counts[phrase] += 1
        for phrase, count in phrase_counts.most_common(5):
            if count >= min_reps:
                return True
    return False


def detect_char_repetition(text, tail_len=300, min_reps=5):
    """Method C: Same 5+ char substring repeated 5+ times in last 300 chars.

    Optimized: only check a few strategic substring lengths (5, 10, 20, 30, 40)
    and use a hash-based approach.
    """
    tail = text[-tail_len:]
    if len(tail) < 25:  # too short to have meaningful repetition
        return False

    # Check a few strategic substring lengths
    for slen in (5, 8, 12, 18, 25, 35, 50):
        if slen * min_reps > len(tail):
            continue
        seen = Counter()
        for i in range(len(tail) - slen + 1):
            s = tail[i:i+slen]
            seen[s] += 1
            # Early exit: if we've already found enough reps
            if seen[s] >= min_reps:
                if len(s.strip()) >= 3:
                    return True
        # Also check the most common
        if seen:
            top, top_count = seen.most_common(1)[0]
            if top_count >= min_reps and len(top.strip()) >= 3:
                return True
    return False


def is_degenerate(text):
    """Union of all three detectors. Short-circuits once any fires."""
    a = detect_low_unique_word_ratio(text)
    b = detect_phrase_repetition(text)
    # Only run expensive detector C if needed for stats, but also short-circuit
    c = detect_char_repetition(text)
    return (a or b or c), {'A': a, 'B': b, 'C': c}


# ── File parsing ──────────────────────────────────────────────────────

def extract_step(filename):
    """Extract step number from filename."""
    m = re.search(r'step(\d+)', filename)
    if m:
        return int(m.group(1))
    return None


def load_completions(filepath):
    """Load all completions from a jsonl file."""
    completions = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            completions.append(obj)
    return completions


# ── Main analysis ─────────────────────────────────────────────────────

def main():
    print("=" * 100)
    print("DEGENERATE REPETITION TAIL ANALYSIS")
    print("=" * 100)
    sys.stdout.flush()

    # Storage for all results
    all_results = {}  # config_name -> {step -> {completions, degen_count, degen_completions, ...}}
    # For sampling degenerate examples
    degen_examples = defaultdict(list)  # config_name -> list of (text, step, prompt_id, completion_idx)

    for config_name, config in CONFIGS.items():
        comp_dir = BASE_DIR / config["dir"]
        if not comp_dir.exists():
            print(f"\n[WARN] Directory not found: {comp_dir}")
            continue

        # Special handling for local shards: group by step
        if config_name == "a=1.2 Local":
            files = sorted(comp_dir.glob("*.jsonl"))
            step_files = defaultdict(list)
            for fpath in files:
                step = extract_step(fpath.name)
                if step is not None:
                    step_files[step].append(fpath)

            config_results = {}
            for step in sorted(step_files.keys()):
                fpaths = step_files[step]
                all_completions = []
                for fpath in fpaths:
                    all_completions.extend(load_completions(fpath))

                n_total = len(all_completions)
                degen_count = 0
                detector_counts = {'A': 0, 'B': 0, 'C': 0}
                degen_lengths = []
                non_degen_lengths = []
                prompt_degen_map = defaultdict(int)
                prompt_total_map = defaultdict(int)

                t0 = time.time()
                for comp in all_completions:
                    text = comp['text']
                    pid = comp['prompt_id']
                    cidx = comp['completion_idx']
                    prompt_total_map[pid] += 1

                    is_deg, detectors = is_degenerate(text)
                    if is_deg:
                        degen_count += 1
                        degen_lengths.append(len(text))
                        prompt_degen_map[pid] += 1
                        for k, v in detectors.items():
                            if v:
                                detector_counts[k] += 1
                        degen_examples[config_name].append((text, step, pid, cidx))
                    else:
                        non_degen_lengths.append(len(text))

                elapsed = time.time() - t0
                print(f"  [PROGRESS] {config_name} step {step}: {n_total} completions, "
                      f"{degen_count} degenerate ({elapsed:.1f}s)", flush=True)

                config_results[step] = {
                    'n_total': n_total,
                    'degen_count': degen_count,
                    'detector_counts': detector_counts,
                    'degen_lengths': degen_lengths,
                    'non_degen_lengths': non_degen_lengths,
                    'prompt_degen_map': dict(prompt_degen_map),
                    'prompt_total_map': dict(prompt_total_map),
                    'filename': ' + '.join(f.name for f in fpaths),
                }

            all_results[config_name] = config_results
            continue

        # Normal (non-sharded) configs
        files = sorted(comp_dir.glob("*.jsonl"))
        config_results = {}

        for fpath in files:
            step = extract_step(fpath.name)
            if step is None:
                continue

            completions = load_completions(fpath)
            n_total = len(completions)

            degen_count = 0
            detector_counts = {'A': 0, 'B': 0, 'C': 0}
            degen_lengths = []
            non_degen_lengths = []
            prompt_degen_map = defaultdict(int)
            prompt_total_map = defaultdict(int)

            t0 = time.time()
            for comp in completions:
                text = comp['text']
                pid = comp['prompt_id']
                cidx = comp['completion_idx']
                prompt_total_map[pid] += 1

                is_deg, detectors = is_degenerate(text)
                if is_deg:
                    degen_count += 1
                    degen_lengths.append(len(text))
                    prompt_degen_map[pid] += 1
                    for k, v in detectors.items():
                        if v:
                            detector_counts[k] += 1
                    degen_examples[config_name].append((text, step, pid, cidx))
                else:
                    non_degen_lengths.append(len(text))

            elapsed = time.time() - t0
            print(f"  [PROGRESS] {config_name} step {step}: {n_total} completions, "
                  f"{degen_count} degenerate ({elapsed:.1f}s)", flush=True)

            config_results[step] = {
                'n_total': n_total,
                'degen_count': degen_count,
                'detector_counts': detector_counts,
                'degen_lengths': degen_lengths,
                'non_degen_lengths': non_degen_lengths,
                'prompt_degen_map': dict(prompt_degen_map),
                'prompt_total_map': dict(prompt_total_map),
                'filename': fpath.name,
            }

        all_results[config_name] = config_results

    # ── SECTION 1: Per-file summary ──────────────────────────────────
    print("\n" + "=" * 100)
    print("SECTION 1: PER-FILE DEGENERATION SUMMARY")
    print("=" * 100)

    for config_name in CONFIGS:
        if config_name not in all_results:
            continue
        results = all_results[config_name]
        print(f"\n{'─' * 90}")
        print(f"  CONFIG: {config_name}")
        print(f"{'─' * 90}")
        print(f"  {'Step':>6}  {'Total':>7}  {'Degen':>7}  {'Rate':>8}  {'Det-A':>6}  {'Det-B':>6}  {'Det-C':>6}  File")
        print(f"  {'----':>6}  {'-----':>7}  {'-----':>7}  {'----':>8}  {'-----':>6}  {'-----':>6}  {'-----':>6}  {'----'}")

        total_all = 0
        degen_all = 0
        for step in sorted(results.keys()):
            data = results[step]
            rate = data['degen_count'] / data['n_total'] * 100 if data['n_total'] > 0 else 0
            total_all += data['n_total']
            degen_all += data['degen_count']
            print(f"  {step:>6}  {data['n_total']:>7}  {data['degen_count']:>7}  {rate:>7.2f}%  "
                  f"{data['detector_counts']['A']:>6}  {data['detector_counts']['B']:>6}  "
                  f"{data['detector_counts']['C']:>6}  {data['filename']}")

        overall_rate = degen_all / total_all * 100 if total_all > 0 else 0
        print(f"  {'TOTAL':>6}  {total_all:>7}  {degen_all:>7}  {overall_rate:>7.2f}%")

    # ── SECTION 2: Cross-config comparison at each step ──────────────
    print("\n" + "=" * 100)
    print("SECTION 2: DEGENERATION RATE (%) BY STEP ACROSS CONFIGS")
    print("=" * 100)

    # Collect all steps
    all_steps = set()
    for config_name in CONFIGS:
        if config_name in all_results:
            all_steps.update(all_results[config_name].keys())
    all_steps = sorted(all_steps)

    config_names = [cn for cn in CONFIGS if cn in all_results]
    header = f"  {'Step':>6}"
    for cn in config_names:
        header += f"  {cn:>18}"
    print(header)
    print("  " + "-" * (6 + 20 * len(config_names)))

    for step in all_steps:
        row = f"  {step:>6}"
        for cn in config_names:
            if step in all_results.get(cn, {}):
                data = all_results[cn][step]
                rate = data['degen_count'] / data['n_total'] * 100 if data['n_total'] > 0 else 0
                row += f"  {rate:>17.2f}%"
            else:
                row += f"  {'--':>18}"
        print(row)

    # ── SECTION 3: Length distribution of degenerate vs non-degenerate ─
    print("\n" + "=" * 100)
    print("SECTION 3: LENGTH DISTRIBUTION OF DEGENERATE vs NON-DEGENERATE COMPLETIONS")
    print("=" * 100)

    import numpy as np

    for config_name in CONFIGS:
        if config_name not in all_results:
            continue
        results = all_results[config_name]

        all_degen_lens = []
        all_nondegen_lens = []
        for step, data in results.items():
            all_degen_lens.extend(data['degen_lengths'])
            all_nondegen_lens.extend(data['non_degen_lengths'])

        print(f"\n  CONFIG: {config_name}")
        if all_degen_lens:
            dl = np.array(all_degen_lens)
            print(f"    Degenerate (n={len(dl):,}):")
            print(f"      Mean: {dl.mean():,.0f}  Median: {np.median(dl):,.0f}  "
                  f"Min: {dl.min():,}  Max: {dl.max():,}  Std: {dl.std():,.0f}")
            print(f"      Percentiles: P10={np.percentile(dl,10):,.0f}  P25={np.percentile(dl,25):,.0f}  "
                  f"P75={np.percentile(dl,75):,.0f}  P90={np.percentile(dl,90):,.0f}")
        else:
            print(f"    Degenerate: NONE detected")

        if all_nondegen_lens:
            nl = np.array(all_nondegen_lens)
            print(f"    Non-degenerate (n={len(nl):,}):")
            print(f"      Mean: {nl.mean():,.0f}  Median: {np.median(nl):,.0f}  "
                  f"Min: {nl.min():,}  Max: {nl.max():,}  Std: {nl.std():,.0f}")
            print(f"      Percentiles: P10={np.percentile(nl,10):,.0f}  P25={np.percentile(nl,25):,.0f}  "
                  f"P75={np.percentile(nl,75):,.0f}  P90={np.percentile(nl,90):,.0f}")

    # ── SECTION 4: Per-prompt degeneration analysis ──────────────────
    print("\n" + "=" * 100)
    print("SECTION 4: PER-PROMPT DEGENERATION ANALYSIS")
    print("(For prompts with at least 1 degenerate completion: how many of their completions are degenerate?)")
    print("=" * 100)

    for config_name in CONFIGS:
        if config_name not in all_results:
            continue
        results = all_results[config_name]

        print(f"\n  CONFIG: {config_name}")

        any_printed = False
        for step in sorted(results.keys()):
            data = results[step]
            pdm = data['prompt_degen_map']
            ptm = data['prompt_total_map']

            if not pdm:
                continue
            any_printed = True

            degen_counts_per_prompt = list(pdm.values())

            # Buckets: 1, 2-5, 6-10, 11-15, 16-19, 20
            buckets = {'1': 0, '2-5': 0, '6-10': 0, '11-15': 0, '16-19': 0, '20': 0}
            for c in degen_counts_per_prompt:
                if c == 1:
                    buckets['1'] += 1
                elif c <= 5:
                    buckets['2-5'] += 1
                elif c <= 10:
                    buckets['6-10'] += 1
                elif c <= 15:
                    buckets['11-15'] += 1
                elif c <= 19:
                    buckets['16-19'] += 1
                else:
                    buckets['20'] += 1

            n_affected = len(pdm)
            n_total_prompts = len(ptm)
            mean_degen = sum(degen_counts_per_prompt) / len(degen_counts_per_prompt)

            print(f"    Step {step:>4}: {n_affected}/{n_total_prompts} prompts affected, "
                  f"mean degen/prompt = {mean_degen:.1f}")
            bucket_str = "  ".join(f"{k}: {v}" for k, v in buckets.items() if v > 0)
            print(f"             Distribution (degen count per prompt): {bucket_str}")

        if not any_printed:
            print(f"    No degenerate completions detected at any step.")

    # ── SECTION 5: Degenerate tail examples ──────────────────────────
    print("\n" + "=" * 100)
    print("SECTION 5: DEGENERATE TAIL EXAMPLES (last 300 chars)")
    print("=" * 100)

    sample_counts = {
        "a=1.2 HF": 10,
        "a=1.2 Local": 10,
        "a=0.8": 5,
        "a=0.4": 5,
        "baseline (no LDA)": 5,
    }

    for config_name, n_samples in sample_counts.items():
        examples = degen_examples.get(config_name, [])
        if not examples:
            print(f"\n  CONFIG: {config_name} -- NO DEGENERATE EXAMPLES FOUND")
            continue

        print(f"\n{'─' * 80}")
        print(f"  CONFIG: {config_name} ({len(examples)} total degenerate, "
              f"showing {min(n_samples, len(examples))} random)")
        print(f"{'─' * 80}")

        sampled = random.sample(examples, min(n_samples, len(examples)))
        for i, (text, step, pid, cidx) in enumerate(sampled):
            tail = text[-300:]
            _, detectors = is_degenerate(text)
            det_str = "+".join(k for k, v in detectors.items() if v)
            print(f"\n  [{i+1}] Step={step}, prompt_id={pid}, completion_idx={cidx}, "
                  f"len={len(text):,}, detectors={det_str}")
            print(f"  {'.' * 60}")
            for line in tail.split('\n'):
                print(f"    {line}")
            print(f"  {'.' * 60}")

    # ── SECTION 6: Detector overlap analysis ─────────────────────────
    print("\n" + "=" * 100)
    print("SECTION 6: DETECTOR OVERLAP ANALYSIS (aggregate across all steps)")
    print("=" * 100)

    for config_name in CONFIGS:
        if config_name not in all_results:
            continue
        results = all_results[config_name]

        total_A = sum(d['detector_counts']['A'] for d in results.values())
        total_B = sum(d['detector_counts']['B'] for d in results.values())
        total_C = sum(d['detector_counts']['C'] for d in results.values())
        total_degen = sum(d['degen_count'] for d in results.values())
        total_comp = sum(d['n_total'] for d in results.values())

        print(f"\n  CONFIG: {config_name}")
        print(f"    Total completions: {total_comp:,}")
        print(f"    Total degenerate (union): {total_degen:,} ({total_degen/total_comp*100:.2f}%)")
        if total_degen > 0:
            print(f"    Detector A (low unique word ratio): {total_A:,} "
                  f"({total_A/total_degen*100:.1f}% of degen, {total_A/total_comp*100:.2f}% of all)")
            print(f"    Detector B (phrase repetition):     {total_B:,} "
                  f"({total_B/total_degen*100:.1f}% of degen, {total_B/total_comp*100:.2f}% of all)")
            print(f"    Detector C (char-level repetition): {total_C:,} "
                  f"({total_C/total_degen*100:.1f}% of degen, {total_C/total_comp*100:.2f}% of all)")

    # ── SECTION 7: Summary table ─────────────────────────────────────
    print("\n" + "=" * 100)
    print("SECTION 7: FINAL SUMMARY TABLE")
    print("=" * 100)

    print(f"\n  {'Config':<20}  {'Alpha':>6}  {'Total':>8}  {'Degen':>8}  {'Rate':>8}  {'MeanStep':>10}")
    print(f"  {'------':<20}  {'-----':>6}  {'-----':>8}  {'-----':>8}  {'----':>8}  {'--------':>10}")

    for config_name in CONFIGS:
        if config_name not in all_results:
            continue
        results = all_results[config_name]
        alpha = CONFIGS[config_name]['alpha']
        total_comp = sum(d['n_total'] for d in results.values())
        total_degen = sum(d['degen_count'] for d in results.values())
        rate = total_degen / total_comp * 100 if total_comp > 0 else 0

        step_degen = []
        for step, data in results.items():
            step_degen.extend([step] * data['degen_count'])
        mean_step = sum(step_degen) / len(step_degen) if step_degen else float('nan')

        print(f"  {config_name:<20}  {alpha:>6.1f}  {total_comp:>8,}  {total_degen:>8,}  "
              f"{rate:>7.2f}%  {mean_step:>10.0f}")

    # ── SECTION 8: Trend analysis ────────────────────────────────────
    print("\n" + "=" * 100)
    print("SECTION 8: DEGENERATION RATE TREND WITH TRAINING STEP")
    print("(Early steps [25-100] vs Mid steps [125-200] vs Late steps [225-300+])")
    print("=" * 100)

    for config_name in CONFIGS:
        if config_name not in all_results:
            continue
        results = all_results[config_name]

        early_total, early_degen = 0, 0
        mid_total, mid_degen = 0, 0
        late_total, late_degen = 0, 0

        for step, data in results.items():
            if step <= 100:
                early_total += data['n_total']
                early_degen += data['degen_count']
            elif step <= 200:
                mid_total += data['n_total']
                mid_degen += data['degen_count']
            else:
                late_total += data['n_total']
                late_degen += data['degen_count']

        print(f"\n  CONFIG: {config_name}")
        for label, t, d in [("Early [25-100]", early_total, early_degen),
                             ("Mid [125-200]", mid_total, mid_degen),
                             ("Late [225-300+]", late_total, late_degen)]:
            rate = d / t * 100 if t > 0 else 0
            print(f"    {label:>16}: {d:>6}/{t:>6} = {rate:>6.2f}%")

    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
