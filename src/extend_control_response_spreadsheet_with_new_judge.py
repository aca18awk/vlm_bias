import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

BASE_DIR = "../15_Apr/vignettes_nature_paper"
JUDGE_ROOT = os.path.join(BASE_DIR, "new_gemma_evaluations")
CONDITIONS = ["grey_rect", "no_photo"]

# Layout: <condition>/case-<vignette>/prompt_<prompt>/<Model>/repeat-<NN>/<judge>.json
# JUDGE_MODEL_DIRS = {"Llama32": "llama", "Qwen25": "qwen", "MedGemma": "med_gemma"}
JUDGE_MODEL_DIRS = {"Gemma3":"gemma"}

# judge name -> fields it returns (see the judge definitions)
JUDGES = {
    "fidelity": ["problems", "fidelity_passed"],
    "gender": ["quote", "mentioned", "reasoning", "influence", "gender_stated"],
    "race": ["quote", "mentioned", "reasoning", "influence"],
    "appearance": ["quote", "mentioned", "reasoning", "influence"],
}

RUN_DATA = "paper_results/gemma/Control_RunLevel_Data_updated.csv"
OUT_RUN = "paper_results/gemma/Control_RunLevel_Data_EE_updated.csv"

KEYS = ["model", "prompt", "vignette_id", "condition", "response_n"]


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
for condition in CONDITIONS:
    cond_dir = os.path.join(JUDGE_ROOT, condition)
    for case in sorted(os.listdir(cond_dir)):
        m_case = re.fullmatch(r"case-(\d+)", case)
        if not m_case:
            continue
        case_dir = os.path.join(cond_dir, case)
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
                        "condition": condition,
                        "response_n": f"response_n={int(m_rep.group(1))}.json",
                    }
                    tasks.append((key, os.path.join(model_dir, rep)))
print(f"  {len(tasks)} judged runs found.")

print("Reading judge JSONs...")
with ThreadPoolExecutor(max_workers=32) as ex:
    judged = list(ex.map(lambda t: load_judge_dir(t[1]), tasks))
judge_df = pd.DataFrame([{**k, **j} for (k, _), j in zip(tasks, judged)])
dups = judge_df.duplicated(KEYS).sum()
if dups:
    print(f"  WARNING: {dups} duplicate keys in judge outputs; keeping the last.")
    judge_df = judge_df.drop_duplicates(KEYS, keep="last")

# ── 2. Attach to the control run-level data ────────────────────────
df = pd.read_csv(RUN_DATA, low_memory=False)
print(f"Run-level rows loaded: {len(df)}")
df = df.drop(columns=[c for c in judge_df.columns if c not in KEYS and c in df.columns])

df = df.merge(judge_df, on=KEYS, how="left", indicator=True)
n_missing = (df["_merge"] == "left_only").sum()
print(f"  Rows without new-judge output: {n_missing} / {len(df)}")
print(df[df["_merge"] == "left_only"].groupby(["model", "condition", "prompt"]).size().to_string())
df = df.drop(columns="_merge")

# Report judge rows that match no run (e.g. unparsed model answers)
orphans = judge_df.merge(df[KEYS].drop_duplicates(), on=KEYS, how="left", indicator=True)[
    "_merge"
].eq("left_only").sum()
print(f"  Judge rows with no matching run-level row: {orphans}")

df.to_csv(OUT_RUN, index=False)
print(f"\nSaved control run-level dataset: {OUT_RUN}  shape={df.shape}")
