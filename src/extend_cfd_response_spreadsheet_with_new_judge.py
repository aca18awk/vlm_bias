import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

BASE_DIR = "../15_Apr/vignettes_nature_paper"
JUDGE_ROOT = os.path.join(BASE_DIR, "new_gemma_evaluations/gemma")
JUDGE_PARTS = [""]  # gemma outputs are not split into parts

# Layout: <part>/<photo_id>/case-<vignette>/prompt_<prompt>/<Model>/repeat-<NN>/<judge>.json
JUDGE_MODEL_DIRS = {"Gemma3":"gemma"}
# # {"Llama32": "llama", "Qwen25": "qwen", "MedGemma": "med_gemma"}

# judge name -> fields it returns (see the judge definitions)
JUDGES = {
    "fidelity": ["problems", "fidelity_passed"],
    "gender": ["quote", "mentioned", "reasoning", "influence", "gender_stated"],
    "race": ["quote", "mentioned", "reasoning", "influence"],
    "appearance": ["quote", "mentioned", "reasoning", "influence"],
}

# Existing run-level spreadsheets (faces only) that the judge columns get attached to
RUN_DATA = [
    "paper_results/gemma/Control_RunLevel_Data_updated.csv"
    # "paper_results/Master_RunLevel_Data_EE.csv",  # llama, qwen
    # "paper_results/med_gemma/Master_RunLevel_Data_Medgemma_EE.csv",  # med_gemma
]
# Columns produced by the old judge; dropped so they don't get mixed up with the new ones
OLD_EE_COLS = [
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

OUT_RUN = "paper_results/gemma/Master_RunLevel_Data_gemma_NewJudge.csv"
OUT_AGG = "paper_results/gemma/Master_Aggregated_Data_gemma_NewJudge.csv"

BOOL_COLS = [
    "gender_mentioned",
    "gender_influence",
    "race_mentioned",
    "race_influence",
    "appearance_mentioned",
    "appearance_influence",
    "fidelity_passed",
]


def load_judge_dir(run_dir):
    """Read the 4 judge files in one repeat dir -> flat dict of prefixed columns."""
    out = {}
    for judge, fields in JUDGES.items():
        path = os.path.join(run_dir, f"{judge}.json")
        data = None
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                data = None
        for field in fields:
            val = data.get(field) if isinstance(data, dict) else None
            if judge == "fidelity" and field == "problems":
                out["fidelity_n_problems"] = len(val) if isinstance(val, list) else None
                val = json.dumps(val) if val is not None else None
            # fidelity_passed already carries its judge name
            col = field if judge == "fidelity" and field != "problems" else f"{judge}_{field}"
            if col == "gender_gender_stated":
                col = "gender_stated"
            out[col] = val
    return out


# ── 1. Walk the judge outputs ──────────────────────────────────────
print("Listing judge output directories...")
tasks = []  # (key dict, run_dir)
for part in JUDGE_PARTS:
    part_dir = os.path.join(JUDGE_ROOT, part)
    for photo_id in sorted(os.listdir(part_dir)):
        photo_dir = os.path.join(part_dir, photo_id)
        if not os.path.isdir(photo_dir):
            continue
        for case in sorted(os.listdir(photo_dir)):
            m_case = re.fullmatch(r"case-(\d+)", case)
            if not m_case:
                continue
            case_dir = os.path.join(photo_dir, case)
            for prompt_dir in sorted(os.listdir(case_dir)):
                if not prompt_dir.startswith("prompt_"):
                    continue
                for judge_model, model in JUDGE_MODEL_DIRS.items():
                    model_dir = os.path.join(case_dir, prompt_dir, judge_model)
                    if not os.path.isdir(model_dir):
                        continue
                    for rep in sorted(os.listdir(model_dir)):
                        m_rep = re.fullmatch(r"repeat-(\d+)", rep)
                        if not m_rep:
                            continue
                        key = {
                            "model": model,
                            "prompt": prompt_dir.replace("prompt_", ""),
                            "vignette_id": int(m_case.group(1)),
                            "photo_id": photo_id,
                            "response_n": f"response_n={int(m_rep.group(1))}.json",
                        }
                        tasks.append((key, os.path.join(model_dir, rep)))
print(f"  {len(tasks)} judged runs found.")

print("Reading judge JSONs...")
with ThreadPoolExecutor(max_workers=32) as ex:
    judged = list(ex.map(lambda t: load_judge_dir(t[1]), tasks))
judge_df = pd.DataFrame([{**k, **j} for (k, _), j in zip(tasks, judged)])
judge_df["judge_version"] = "new"
dups = judge_df.duplicated(["model", "prompt", "vignette_id", "photo_id", "response_n"]).sum()
if dups:
    print(f"  WARNING: {dups} duplicate keys in judge outputs; keeping the last.")
    judge_df = judge_df.drop_duplicates(
        ["model", "prompt", "vignette_id", "photo_id", "response_n"], keep="last"
    )

# ── 2. Attach to the run-level data ────────────────────────────────
df = pd.concat([pd.read_csv(p, low_memory=False) for p in RUN_DATA], ignore_index=True)
print(f"Run-level rows loaded: {len(df)}")
df = df.drop(columns=[c for c in OLD_EE_COLS if c in df.columns] + ["run_id"], errors="ignore")

keys = ["model", "prompt", "vignette_id", "photo_id", "response_n"]
df = df.merge(judge_df.drop(columns=["judge_version"]), on=keys, how="left", indicator=True)
n_missing = (df["_merge"] == "left_only").sum()
print(f"  Rows without new-judge output: {n_missing} / {len(df)}")
print(df[df["_merge"] == "left_only"].groupby(["model", "prompt"]).size().to_string())
df = df.drop(columns="_merge")
df.insert(0, "run_id", range(1, len(df) + 1))

# Report judge rows that match no run (e.g. unparsed model answers)
orphans = judge_df.merge(
    df[keys].drop_duplicates(), on=keys, how="left", indicator=True
)["_merge"].eq("left_only").sum()
print(f"  Judge rows with no matching run-level row: {orphans}")

df.to_csv(OUT_RUN, index=False)
print(f"\nSaved run-level dataset: {OUT_RUN}  shape={df.shape}")

# # ── 3. Aggregated version (one row per model/prompt/vignette/face) ──
# agg_path = "paper_results/gemma/Master_Aggregated_Data_EE.csv"
# # agg_med_path = "paper_results/med_gemma/Master_Aggregated_Data_Medgemma_EE.csv"
# df_agg = pd.concat(
#     [pd.read_csv(agg_path), pd.read_csv(agg_med_path)], ignore_index=True
# )
# df_agg = df_agg.drop(
#     columns=[c for c in df_agg.columns if c.endswith("_rate") and c != "admit_rate"]
#     + [c for c in df_agg.columns if c.endswith("_n") and c != "n_total"],
#     errors="ignore",
# )

# num = df.copy()
# for col in BOOL_COLS:
#     num[col] = pd.to_numeric(
#         num[col].map({True: 1, False: 0, "True": 1, "False": 0}), errors="coerce"
#     )
# cell_keys = ["model", "prompt", "vignette_id", "photo_id"]
# grouped = num.groupby(cell_keys)[BOOL_COLS]
# rates = grouped.mean().rename(columns={c: f"{c}_rate" for c in BOOL_COLS})
# counts = grouped.count().rename(columns={c: f"{c}_n" for c in BOOL_COLS})
# summary = pd.concat([rates, counts], axis=1).reset_index()
# summary = summary[cell_keys + [c for b in BOOL_COLS for c in (f"{b}_rate", f"{b}_n")]]

# df_agg = df_agg.merge(summary, on=cell_keys, how="left")
# df_agg.to_csv(OUT_AGG, index=False)
# print(f"Saved aggregated dataset: {OUT_AGG}  shape={df_agg.shape}")
# print(df_agg[["model", "prompt", "photo_id"] + [f"{c}_rate" for c in BOOL_COLS]].head().to_string())
