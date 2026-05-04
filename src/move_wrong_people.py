import os
import json
import shutil

BASE = os.path.join(os.path.dirname(__file__), "../15_Apr/vignettes_nature_paper")
BASE = os.path.abspath(BASE)

MAPPING_FILE = os.path.join(BASE, "ground_truth_mapping.json")

MODELS = {
    "qwen25":  "evaluations_qwen25_cfd",
    "llama32": "evaluations_llama32_cfd",
}

PROMPTS = ["prompt_baseline", "prompt_ignore", "prompt_acknowledge"]

FACE_SLOTS = ["BM_4", "IM_1", "IM_3", "IM_5", "IF_1", "IF_2", "IF_3", "IF_4"]

VIGNETTES = [f"{v:02d}" for v in range(1, 13)]

with open(MAPPING_FILE) as f:
    mapping = json.load(f)

moved = 0
skipped = 0
missing_key = 0

for model_key, evaluations_dir in MODELS.items():
    wrong_dir = f"wrong_people_evaluations_{model_key}"

    for prompt in PROMPTS:
        for vignette in VIGNETTES:
            for face_slot in FACE_SLOTS:
                key = f"{vignette}_{face_slot}"
                encoded = mapping.get(key)

                if not encoded:
                    print(f"WARNING: no mapping entry for {key}")
                    missing_key += 1
                    continue

                src = os.path.join(BASE, evaluations_dir, prompt, encoded)
                dst = os.path.join(BASE, wrong_dir, prompt, encoded)

                if not os.path.isdir(src):
                    print(f"SKIP (not found): {os.path.relpath(src, BASE)}")
                    skipped += 1
                    continue

                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.move(src, dst)
                print(f"MOVED: {os.path.relpath(src, BASE)}  →  {wrong_dir}/{prompt}/{encoded}")
                moved += 1

print(f"\nEvaluations done. Moved: {moved}  |  Skipped: {skipped}  |  Missing keys: {missing_key}")

# ── Assets: pdf_cfd, jpg_cfd, docs_cfd ─────────────────────────────
# pdf and jpg use encoded filenames; docs use the human-readable face slot name.

ASSET_DIRS = [
    ("pdf_cfd",  "wrong_people_pdf",  "pdf"),
    ("jpg_cfd",  "wrong_people_jpg",  "jpg"),
]

moved = skipped = missing_key = 0

for src_dir, dst_dir, ext in ASSET_DIRS:
    for vignette in VIGNETTES:
        for face_slot in FACE_SLOTS:
            key = f"{vignette}_{face_slot}"
            encoded = mapping.get(key)

            if not encoded:
                print(f"WARNING: no mapping entry for {key}")
                missing_key += 1
                continue

            src = os.path.join(BASE, src_dir, f"{encoded}.{ext}")
            dst = os.path.join(BASE, dst_dir, f"{encoded}.{ext}")

            if not os.path.isfile(src):
                print(f"SKIP (not found): {src_dir}/{encoded}.{ext}")
                skipped += 1
                continue

            os.makedirs(os.path.join(BASE, dst_dir), exist_ok=True)
            shutil.move(src, dst)
            print(f"MOVED: {src_dir}/{encoded}.{ext}  →  {dst_dir}/")
            moved += 1

# docs use the face slot name directly (e.g. 01_BM_4.docx)
for vignette in VIGNETTES:
    for face_slot in FACE_SLOTS:
        filename = f"{vignette}_{face_slot}.docx"
        src = os.path.join(BASE, "docs_cfd", filename)
        dst = os.path.join(BASE, "wrong_people_docs", filename)

        if not os.path.isfile(src):
            print(f"SKIP (not found): docs_cfd/{filename}")
            skipped += 1
            continue

        os.makedirs(os.path.join(BASE, "wrong_people_docs"), exist_ok=True)
        shutil.move(src, dst)
        print(f"MOVED: docs_cfd/{filename}  →  wrong_people_docs/")
        moved += 1

print(f"\nAssets done. Moved: {moved}  |  Skipped: {skipped}  |  Missing keys: {missing_key}")
print(f"Expected: 12 vignettes × {len(FACE_SLOTS)} slots × 3 asset types = {12*len(FACE_SLOTS)*3}")
