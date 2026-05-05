import json
import os
import re

import pandas as pd

BASE_DIR = "../15_Apr/vignettes_nature_paper"
MAPPING_FILE = os.path.join(BASE_DIR, "ground_truth_mapping.json")

# eval dir suffix = the JUDGE model, not the subject
# evaluations_qwen25_cfd = qwen judged llama's responses
# evaluations_llama32_cfd = llama judged qwen's responses
MODEL_SUFFIX = {"llama": "qwen25", "qwen": "llama32"}

with open(MAPPING_FILE) as f:
    mapping = json.load(f)  # human_name -> encoded_name

print("Loading Master_RunLevel_Data.csv...")
df = pd.read_csv("Master_RunLevel_Data.csv")
print(f"  {len(df)} rows loaded.")

ee_cols = [
    "mentions_race",
    "evidence_race",
    "mentions_gender",
    "evidence_gender",
    "gender_used_in_reasoning",
    "evidence_gender_reasoning",
    "clinical_fidelity_passed",
    "evidence_hallucination",
    "fidelity_confidence",
]

bool_ee_cols = [
    "mentions_race",
    "mentions_gender",
    "gender_used_in_reasoning",
    "clinical_fidelity_passed",
]

missing_evals = 0
ee_records = []

for _, row in df.iterrows():
    model = row["model"]
    prompt = row["prompt"]
    vignette_id = int(row["vignette_id"])
    photo_id = row["photo_id"]  # e.g. "LM_5"
    response_n = row["response_n"]  # e.g. "response_n=0.json"

    # Reconstruct human_name and look up encoded_name
    human_name = f"{vignette_id:02d}_{photo_id}"
    encoded_name = mapping.get(human_name)

    # Extract run index from response_n filename
    m = re.search(r"response_n=(\d+)\.json", response_n)
    run_idx = m.group(1) if m else None

    eval_result = {col: None for col in ee_cols}

    if encoded_name and run_idx is not None:
        suffix = MODEL_SUFFIX.get(model)
        eval_path = os.path.join(
            BASE_DIR,
            f"evaluations_{suffix}_cfd",
            f"prompt_{prompt}",
            encoded_name,
            f"eval_n={run_idx}.json",
        )
        if os.path.exists(eval_path):
            try:
                with open(eval_path) as f:
                    eval_result = json.load(f)
            except json.JSONDecodeError:
                missing_evals += 1
        else:
            missing_evals += 1
    else:
        missing_evals += 1

    ee_records.append(eval_result)

print(f"  Missing eval files: {missing_evals} / {len(df)}")

ee_df = pd.DataFrame(ee_records)

# Add sequential run_id and attach EE columns
df.insert(0, "run_id", range(1, len(df) + 1))
df = pd.concat([df, ee_df], axis=1)

out_path = "Master_RunLevel_Data_EE.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved extended run-level dataset: {out_path}")
print(f"Shape: {df.shape[0]} rows × {df.shape[1]} cols")
print("\nHead (key columns):")
key_cols = ["run_id", "model", "prompt", "vignette_id", "photo_id", "admit_decision",
            "mentions_race", "mentions_gender", "gender_used_in_reasoning",
            "clinical_fidelity_passed", "evidence_hallucination"]
print(df[key_cols].head().to_string())

# ── Extend Master_Aggregated_Data.csv ─────────────────────────────────────────
print("\nLoading Master_Aggregated_Data.csv...")
df_agg = pd.read_csv("Master_Aggregated_Data.csv")
print(f"  {len(df_agg)} rows loaded.")

# Coerce bool EE cols to numeric so mean() works even with None values
for col in bool_ee_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

cell_keys = ["model", "prompt", "vignette_id", "photo_id"]
grouped = df.groupby(cell_keys)[bool_ee_cols]
ee_rates = grouped.mean().rename(columns={c: f"{c}_rate" for c in bool_ee_cols})
ee_counts = grouped.count().rename(columns={c: f"{c}_n" for c in bool_ee_cols})
ee_summary = pd.concat([ee_rates, ee_counts], axis=1).reset_index()

# Interleave rate and n columns for readability: mentions_race_rate, mentions_race_n, ...
ordered_cols = cell_keys + [col for c in bool_ee_cols for col in (f"{c}_rate", f"{c}_n")]
ee_summary = ee_summary[ordered_cols]

df_agg = df_agg.merge(ee_summary, on=cell_keys, how="left")

agg_out_path = "Master_Aggregated_Data_EE.csv"
df_agg.to_csv(agg_out_path, index=False)
print(f"\nSaved extended aggregated dataset: {agg_out_path}")
print(f"Shape: {df_agg.shape[0]} rows × {df_agg.shape[1]} cols")
print("\nNew EE columns added:")
ee_new_cols = [c for c in df_agg.columns if c.endswith("_rate") or c.endswith("_n")]
print(df_agg[["model", "prompt", "photo_id"] + ee_new_cols].head().to_string())
