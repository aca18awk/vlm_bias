import json
import os
import sys
from collections import defaultdict

MAPPING_FILE = "../15_Apr/vignettes_nature_paper/ground_truth_mapping.json"
PROMPTS = ["prompt_baseline", "prompt_ignore", "prompt_acknowledge"]
N_COMPLETIONS = 20

RUNS = [
    {
        "label": "results_med_gemma",
        "dir": "../15_Apr/vignettes_nature_paper/results_med_gemma",
        # The 12 no-photo vignettes aren't in jpg_cfd; they're being run
        # separately and saved under results_med_gemma_control instead.
        "no_photo_dir": "../15_Apr/vignettes_nature_paper/results_med_gemma_control",
    },
    {
        "label": "evaluations_for_med_from_qwen25_cfd",
        "dir": "../15_Apr/vignettes_nature_paper/evaluations_for_med_from_qwen25_cfd",
        "no_photo_dir": "../15_Apr/vignettes_nature_paper/evaluations_for_med_from_qwen25_control",
    },
]

with open(MAPPING_FILE) as f:
    mapping = json.load(f)

encoded_names = list(mapping.values())
total_vignettes = len(encoded_names)
total_expected = total_vignettes * len(PROMPTS) * N_COMPLETIONS

# Vignettes with no CFD photo (e.g. "01_no_photo" -> "01_859") live outside jpg_cfd.
no_photo_ids = {v for k, v in mapping.items() if "no_photo" in k}


def count_dir(results_dir, vignette_ids):
    counts = defaultdict(lambda: defaultdict(int))
    for prompt in PROMPTS:
        prompt_dir = os.path.join(results_dir, prompt)
        if not os.path.isdir(prompt_dir):
            continue
        for vignette in os.listdir(prompt_dir):
            if vignette_ids is not None and vignette not in vignette_ids:
                continue
            vdir = os.path.join(prompt_dir, vignette)
            if os.path.isdir(vdir):
                counts[vignette][prompt] = len([f for f in os.listdir(vdir) if f.endswith(".json")])
    return counts


def check_progress(label, results_dir, no_photo_dir=None):
    print("#" * 60)
    print(f"# {label}")
    print("#" * 60)

    # Count files per vignette per prompt
    # done[encoded_name][prompt] = count of response files
    main_ids = None if not no_photo_dir else (set(encoded_names) - no_photo_ids)
    done = count_dir(results_dir, main_ids)

    if no_photo_dir:
        print(f"(no-photo vignettes counted separately from: {no_photo_dir})")
        no_photo_done = count_dir(no_photo_dir, no_photo_ids)
        for v, prompts in no_photo_done.items():
            for p, c in prompts.items():
                done[v][p] = c

    total_done = sum(sum(p.values()) for p in done.values())

    print("=" * 60)
    print(f"OVERALL: {total_done:,} / {total_expected:,} files  ({100*total_done/total_expected:.1f}%)")
    print("=" * 60)

    # Per-prompt breakdown
    print("\nPer prompt:")
    for prompt in PROMPTS:
        n = sum(done[v][prompt] for v in encoded_names)
        exp = total_vignettes * N_COMPLETIONS
        print(f"  {prompt:<22} {n:>6,} / {exp:,}  ({100*n/exp:.1f}%)")

    # Vignettes with 0 files across ALL prompts (never touched)
    never_started = [v for v in encoded_names if all(done[v][p] == 0 for p in PROMPTS)]
    print(f"\nVignettes never started: {len(never_started)} / {total_vignettes}")
    if never_started:
        print("  IDs:", ", ".join(sorted(never_started)))

    # Partially complete vignettes (started but not all N_COMPLETIONS runs for some prompt)
    partial = []
    for v in encoded_names:
        for p in PROMPTS:
            c = done[v][p]
            if 0 < c < N_COMPLETIONS:
                partial.append((v, p, c))
    print(f"\nPartially complete (in progress or interrupted): {len(partial)} vignette x prompt pairs")
    if partial:
        for v, p, c in partial:
            print(f"  {v}  {p}  {c}/{N_COMPLETIONS}")

    # Any vignette that is not fully done (never started OR partial in any prompt)
    incomplete_ids = sorted({v for v in encoded_names if any(done[v][p] < N_COMPLETIONS for p in PROMPTS)})
    print(f"\nIncomplete or not-started vignettes (any prompt < {N_COMPLETIONS}): {len(incomplete_ids)} / {total_vignettes}")
    if incomplete_ids:
        print("  IDs:", ", ".join(incomplete_ids))

    print()


if __name__ == "__main__":
    for run in RUNS:
        check_progress(run["label"], run["dir"], run.get("no_photo_dir"))
