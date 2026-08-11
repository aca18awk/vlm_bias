import os
import json
import re

import pandas as pd

# ── Paths & Setup ──
BASE_DIR = "../15_Apr/vignettes_nature_paper"

CFD_MODEL_DIRS = {
    "llama": os.path.join(BASE_DIR, "results_llama32_cfd"),
    "qwen":  os.path.join(BASE_DIR, "results_qwen25_cfd"),
}
CONTROL_MODEL_DIRS = {
    "llama":     os.path.join(BASE_DIR, "results_llama32_control"),
    "qwen":      os.path.join(BASE_DIR, "results_qwen25_control"),
    # med_gemma's control dir holds BOTH no_photo and grey_rect runs already,
    # so it's used for both conditions below (no separate CFD dir needed).
    "med_gemma": os.path.join(BASE_DIR, "results_med_gemma_control"),
    "gemma":  os.path.join(BASE_DIR, "results_gemma3_control"),
    
}
CFD_MAPPING_FILE     = os.path.join(BASE_DIR, "ground_truth_mapping.json")
CONTROL_MAPPING_FILE = os.path.join(BASE_DIR, "ground_truth_mapping_control.json")

PROMPT_TYPES = ["prompt_baseline", "prompt_ignore", "prompt_acknowledge"]

BORDERLINE_IDS = {
    "qwen":      [3, 8, 9, 11, 12],
    "llama":     [3, 8, 9, 11, 12, 2, 6, 7, 1],
    "med_gemma": [1, 2, 3, 6, 7, 8, 9, 11, 12],
    "gemma": [1, 2, 3, 6, 7, 8, 9, 11, 12]
}

# eval dir template = evaluations_{...}_{DOMAIN}
EVAL_DIR_TEMPLATE = {
    "llama":     "evaluations_qwen25_{domain}",
    "qwen":      "evaluations_llama32_{domain}",
    "med_gemma": "evaluations_for_med_from_qwen25_{domain}",
    "gemma": "evaluations_for_gemma_from_qwen25_control"
}
# Domain each (model, condition) pair's judge evaluations live under.
# llama/qwen: judged separately per domain (cfd for no_photo, control for grey_rect).
# med_gemma: both conditions were judged together in a single "control" pass
# (evaluations_for_med_from_qwen25_cfd exists but is empty for no_photo).
DOMAIN_FOR = {
    ("llama", "no_photo"):      "cfd",
    ("llama", "grey_rect"):     "control",
    ("qwen", "no_photo"):       "cfd",
    ("qwen", "grey_rect"):      "control",
    ("med_gemma", "no_photo"):  "control",
    ("med_gemma", "grey_rect"): "control",
    ("gemma", "no_photo"):  "control",
    ("gemma", "grey_rect"): "control",

}

BOOL_EE_COLS = [
    "mentions_race",
    "mentions_gender",
    "gender_used_in_reasoning",
    "clinical_fidelity_passed",
]
ALL_EE_COLS = BOOL_EE_COLS + [
    "evidence_race",
    "evidence_gender",
    "evidence_gender_reasoning",
    "evidence_hallucination",
    "fidelity_confidence",
]


def load_maps(path):
    with open(path) as f:
        mapping = json.load(f)  # human_name -> encoded_name
    inv_map = {v: k for k, v in mapping.items()}
    return mapping, inv_map


def parse_runs(model_dirs, inv_map, condition_key, model_name):
    """Parse all run-level JSON files for a given condition ('no_photo' or 'grey_rect')."""
    rows = []
    borderline_ids = BORDERLINE_IDS[model_name]

    for model_name_, results_dir in model_dirs.items():
        if model_name_ != model_name:
            continue
        if not os.path.isdir(results_dir):
            print(f"  Warning: directory not found: {results_dir}")
            continue

        for prompt_dir_name in PROMPT_TYPES:
            prompt_path = os.path.join(results_dir, prompt_dir_name)
            if not os.path.isdir(prompt_path):
                continue
            clean_prompt = prompt_dir_name.replace("prompt_", "")

            for encoded_name in os.listdir(prompt_path):
                human_name = inv_map.get(encoded_name)
                if not human_name:
                    continue

                # human_name format: "01_no_photo" or "01_grey_rect"
                parts = human_name.split("_", 1)
                if len(parts) != 2:
                    continue
                vignette_str, condition = parts
                if condition != condition_key:
                    continue

                vignette_id = int(vignette_str)
                vignette_class = "borderline" if vignette_id in borderline_ids else "clear"

                run_dir = os.path.join(prompt_path, encoded_name)
                for file in sorted(f for f in os.listdir(run_dir) if f.endswith(".json")):
                    try:
                        with open(os.path.join(run_dir, file)) as f:
                            data = json.load(f)
                    except json.JSONDecodeError:
                        continue

                    ans = data.get("parsed_answer", "")
                    if ans in ["admit", "discharge", "**admit**", "**discharge**", "*admit*", "*discharge*"]:
                        rows.append({
                            "model":          model_name,
                            "prompt":         clean_prompt,
                            "vignette_id":    vignette_id,
                            "vignette_class": vignette_class,
                            "condition":      condition_key,
                            "encoded_name":   encoded_name,
                            "response_n":     file,
                            "admit_decision": 1 if ans in ["admit", "**admit**", "*admit*"] else 0,
                        })
    return rows


# ── 1. Load mappings ──────────────────────────────────────────────────
_, inv_map_cfd     = load_maps(CFD_MAPPING_FILE)
_, inv_map_control = load_maps(CONTROL_MAPPING_FILE)

# ── 2. Collect run-level data ─────────────────────────────────────────
master_rows = []
for model_name in ["gemma"]:
    print(f"Parsing no_photo for {model_name}...")
    if model_name in CFD_MODEL_DIRS:
        rows = parse_runs(CFD_MODEL_DIRS, inv_map_cfd, "no_photo", model_name)
    else:
        # med_gemma's no_photo runs live alongside grey_rect in its control dir
        rows = parse_runs(CONTROL_MODEL_DIRS, inv_map_control, "no_photo", model_name)
    print(f"  → {len(rows)} rows")
    master_rows.extend(rows)

    print(f"Parsing grey_rect for {model_name}...")
    rows = parse_runs(CONTROL_MODEL_DIRS, inv_map_control, "grey_rect", model_name)
    print(f"  → {len(rows)} rows")
    master_rows.extend(rows)

df_run_level = pd.DataFrame(master_rows)
print(f"\nRun-level dataframe: {df_run_level.shape[0]} rows × {df_run_level.shape[1]} cols")

# ── 3. Join EE evaluation data ────────────────────────────────────────
print("Joining EE evaluation data...")
missing_evals = 0
ee_records = []

for _, row in df_run_level.iterrows():
    model        = row["model"]
    condition    = row["condition"]
    prompt       = row["prompt"]
    encoded_name = row["encoded_name"]
    response_n   = row["response_n"]

    m = re.search(r"response_n=(\d+)\.json", response_n)
    run_idx = m.group(1) if m else None

    eval_result = {col: None for col in ALL_EE_COLS}

    if run_idx is not None:
        eval_dir = EVAL_DIR_TEMPLATE[model].format(domain=DOMAIN_FOR[(model, condition)])
        eval_path = os.path.join(
            BASE_DIR,
            eval_dir,
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

print(f"  Missing eval files: {missing_evals} / {len(df_run_level)}")

ee_df = pd.DataFrame(ee_records)
df_run_level = pd.concat([df_run_level.reset_index(drop=True), ee_df], axis=1)

# ── 4. Aggregate ──────────────────────────────────────────────────────
group_cols = ["model", "prompt", "vignette_id", "vignette_class", "condition"]

df_aggregated = df_run_level.groupby(group_cols).agg(
    n_admits=("admit_decision", "sum"),
    n_total=("admit_decision", "count"),
).reset_index()

df_aggregated["admit_rate"] = df_aggregated["n_admits"] / df_aggregated["n_total"]

# Coerce bool EE cols to numeric then aggregate rate + count per cell
for col in BOOL_EE_COLS:
    df_run_level[col] = pd.to_numeric(df_run_level[col], errors="coerce")

grouped = df_run_level.groupby(group_cols)[BOOL_EE_COLS]
ee_rates  = grouped.mean().rename(columns={c: f"{c}_rate" for c in BOOL_EE_COLS})
ee_counts = grouped.count().rename(columns={c: f"{c}_n"    for c in BOOL_EE_COLS})
ee_summary = pd.concat([ee_rates, ee_counts], axis=1).reset_index()

# Interleave rate and n columns: mentions_race_rate, mentions_race_n, ...
ordered_ee = [col for c in BOOL_EE_COLS for col in (f"{c}_rate", f"{c}_n")]
ee_summary = ee_summary[group_cols + ordered_ee]

df_aggregated = df_aggregated.merge(ee_summary, on=group_cols, how="left")

# ── 5. Save ───────────────────────────────────────────────────────────
# Drop encoded_name from run-level output (internal lookup key only)
df_run_level.drop(columns=["encoded_name"], inplace=True)

df_run_level.to_csv("Control_RunLevel_Data_EE_gemma.csv", index=False)
df_aggregated.to_csv("Control_Aggregated_Data_EE_gemma.csv", index=False)

print(f"\nDone!")
print(f"Run-level:  {len(df_run_level)} rows × {df_run_level.shape[1]} cols")
print(f"Aggregated: {len(df_aggregated)} rows × {df_aggregated.shape[1]} cols")

print("\n── Diagnostics ──")
incomplete = df_aggregated[df_aggregated["n_total"] < 20]
print(f"Cells with < 20 runs: {len(incomplete)}" + (" ← WARNING" if len(incomplete) else " ✓"))
if len(incomplete):
    print(incomplete[["model", "prompt", "condition", "vignette_id", "n_total"]].to_string())

print(f"\nn_total: min={df_aggregated['n_total'].min()}  max={df_aggregated['n_total'].max()}  mean={df_aggregated['n_total'].mean():.1f}")
print("\nHead:")
print(df_aggregated.head(10).to_string())
print("\nCell sizes (model × prompt × condition):")
print(df_aggregated.groupby(["model", "prompt", "condition"]).size().to_string())
