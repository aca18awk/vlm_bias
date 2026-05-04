import os
import json
import shutil

BASE = os.path.join(os.path.dirname(__file__), "../15_Apr/vignettes_nature_paper")
BASE = os.path.abspath(BASE)

MAPPING_FILE     = os.path.join(BASE, "ground_truth_mapping.json")
NEW_INDIAN_FILE  = os.path.join(BASE, "ground_truth_mapping_control.json")

PROMPTS = ["prompt_baseline", "prompt_ignore", "prompt_acknowledge"]

# ── 1. Uniqueness check ─────────────────────────────────────────────
with open(MAPPING_FILE) as f:
    mapping = json.load(f)

values = list(mapping.values())
if len(values) != len(set(values)):
    from collections import Counter
    dupes = [v for v, n in Counter(values).items() if n > 1]
    print(f"ABORT: {len(dupes)} duplicate encoded ID(s) found in ground_truth_mapping.json:")
    for d in dupes:
        keys = [k for k, v in mapping.items() if v == d]
        print(f"  {d}  →  {keys}")
    raise SystemExit(1)

print(f"Uniqueness check passed: all {len(values)} IDs are unique.\n")

# # ── 2. Move new_indian folders into _cfd ────────────────────────────
# # Source → destination directory pairs (skip if source doesn't exist)
# DIR_PAIRS = [
#     # ("results_qwen25_new_indian",     "results_qwen25_cfd"),
#     # ("results_llama32_new_indian",    "results_llama32_cfd"),
#     ("evaluations_llama32_new_indian","evaluations_llama32_cfd"),
# ]

# with open(NEW_INDIAN_FILE) as f:
#     new_indian = json.load(f)

# encoded_ids = sorted(set(new_indian.values()))

# moved = skipped = 0

# for src_dir_name, dst_dir_name in DIR_PAIRS:
#     src_base = os.path.join(BASE, src_dir_name)
#     dst_base = os.path.join(BASE, dst_dir_name)

#     if not os.path.isdir(src_base):
#         print(f"SKIP (dir not found): {src_dir_name}")
#         continue

#     print(f"\n{src_dir_name}  →  {dst_dir_name}")

#     for prompt in PROMPTS:
#         for encoded in encoded_ids:
#             src = os.path.join(src_base, prompt, encoded)
#             dst = os.path.join(dst_base, prompt, encoded)

#             if not os.path.isdir(src):
#                 print(f"  SKIP (not found): {prompt}/{encoded}")
#                 skipped += 1
#                 continue

#             os.makedirs(os.path.join(dst_base, prompt), exist_ok=True)
#             shutil.move(src, dst)
#             print(f"  MOVED: {prompt}/{encoded}")
#             moved += 1

# print(f"\nResults/evaluations done. Moved: {moved}  |  Skipped: {skipped}")
# print(f"Expected: {len(encoded_ids)} IDs × {len(PROMPTS)} prompts × {len(DIR_PAIRS)} dirs = {len(encoded_ids)*len(PROMPTS)*len(DIR_PAIRS)}")
