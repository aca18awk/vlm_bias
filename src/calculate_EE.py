import json
import os

import pandas as pd
from scipy.stats import chi2_contingency, pointbiserialr

EVAL_DIR = "../15_Apr/vignettes_nature_paper/evaluations_qwen25_cfd
MAPPING_FILE = "../15_Apr/vignettes_nature_paper/ground_truth_mapping.json"

# 1. Load the reverse mapping (Encoded Name -> Human Name)
with open(MAPPING_FILE, "r") as f:
    mapping = json.load(f)
inv_map = {v: k for k, v in mapping.items()}

# 2. Build the Master DataFrame
rows = []
for prompt_type in os.listdir(EVAL_DIR):
    prompt_dir = os.path.join(EVAL_DIR, prompt_type)
    if not os.path.isdir(prompt_dir):
        continue

    for encoded_name in os.listdir(prompt_dir):
        human_name = inv_map.get(encoded_name)
        if not human_name:
            continue

        # Parse the subgroup from the human name (e.g., "01_black_man")
        parts = human_name.split("_")
        vignette_id = parts[0]
        if len(parts) == 3:
            subgroup = f"{parts[1]}_{parts[2]}"  # e.g., "black_man"
        else:
            subgroup = "no_photo"

        run_dir = os.path.join(prompt_dir, encoded_name)
        for file in os.listdir(run_dir):
            if not file.endswith(".json"):
                continue

            with open(os.path.join(run_dir, file), "r") as f:
                data = json.load(f)

            # Skip runs that threw API errors (nulls)
            if data.get("mentions_race") is None:
                continue

            rows.append(
                {
                    "prompt_type": prompt_type,
                    "vignette_id": vignette_id,
                    "subgroup": subgroup,
                    "mentions_race": bool(data.get("mentions_race")),
                    "mentions_gender": bool(data.get("mentions_gender")),
                    "clinical_fidelity_passed": bool(
                        data.get("clinical_fidelity_passed")
                    ),
                }
            )

df = pd.DataFrame(rows)
# Create our specific Hallucination column (Inverse of fidelity)
df["hallucinated"] = ~df["clinical_fidelity_passed"]

print(f"Successfully loaded {len(df)} valid evaluation runs.\n")


# ---------------------------------------------------------
# Helper Function for Printing Clean Tables
# ---------------------------------------------------------
def print_stats_table(df_subset, title):
    print(f"--- {title} ---")
    if df_subset.empty:
        print("No data available.\n")
        return

    stats = (
        df_subset.groupby("subgroup")[
            ["hallucinated", "mentions_race", "mentions_gender"]
        ]
        .mean()
        .reset_index()
    )

    stats = stats.rename(
        columns={
            "hallucinated": "P(Hallucination)",
            "mentions_race": "EIE: P(Race in Exp)",
            "mentions_gender": "EIE: P(Gender in Exp)",
        }
    )

    # Convert to percentages for readability
    stats["P(Hallucination)"] = (stats["P(Hallucination)"] * 100).round(1).astype(
        str
    ) + "%"
    stats["EIE: P(Race in Exp)"] = (stats["EIE: P(Race in Exp)"] * 100).round(1).astype(
        str
    ) + "%"
    stats["EIE: P(Gender in Exp)"] = (stats["EIE: P(Gender in Exp)"] * 100).round(
        1
    ).astype(str) + "%"

    print(stats.to_string(index=False))
    print("\n")


# ---------------------------------------------------------
# 3. Print the Tables (Aggregated + Cases)
# ---------------------------------------------------------
prompt_types = ["prompt_baseline", "prompt_ignore"]

print("=========================================================")
print("                 AGGREGATED STATISTICS                   ")
print("=========================================================\n")
for pt in prompt_types:
    pt_df = df[df["prompt_type"] == pt]
    print_stats_table(pt_df, f"OVERALL AGGREGATED - {pt.upper()}")

print("=========================================================")
print("                 CASE-BY-CASE STATISTICS                 ")
print("=========================================================\n")
for vignette_id in sorted(df["vignette_id"].unique()):
    print(f"################ CASE {vignette_id} ################")
    case_df = df[df["vignette_id"] == vignette_id]

    for pt in prompt_types:
        pt_case_df = case_df[case_df["prompt_type"] == pt]
        print_stats_table(pt_case_df, f"CASE {vignette_id} - {pt.upper()}")


# ---------------------------------------------------------
# 4. Correlation Analysis (Split by Prompt)
# ---------------------------------------------------------
print("=========================================================")
print("                 CORRELATION ANALYSIS                    ")
print("=========================================================\n")

for pt in prompt_types:
    print(f"--- GLOBAL CORRELATION: {pt.upper()} ---")
    pt_df = df[df["prompt_type"] == pt]

    corr_gender = pt_df["hallucinated"].corr(pt_df["mentions_gender"])
    corr_race = pt_df["hallucinated"].corr(pt_df["mentions_race"])

    print(
        f"Correlation between Hallucinating and Mentioning Gender: {corr_gender:.3f}"
        if pd.notna(corr_gender)
        else "Correlation Gender: NaN (Zero variance)"
    )
    print(
        f"Correlation between Hallucinating and Mentioning Race:   {corr_race:.3f}"
        if pd.notna(corr_race)
        else "Correlation Race: NaN (Zero variance)"
    )

    if pd.isna(corr_gender):
        print(
            "-> Interpretation: Could not calculate gender correlation (model likely behaved identically across all runs)."
        )
    elif corr_gender < 0:
        print(
            "-> Interpretation: Negative correlation. When the VLM mentions the patient's gender, it is LESS likely to hallucinate."
        )
    else:
        print(
            "-> Interpretation: Positive correlation. When the VLM mentions the patient's gender, it is MORE likely to hallucinate."
        )
    print("")

# ---------------------------------------------------------
# 5. Gender Mention Correlation (is_male vs mentions_gender)
# ---------------------------------------------------------

print("=========================================================")
print("         GENDER MENTION CORRELATION ANALYSIS             ")
print("=========================================================\n")

df_photo = df[df["subgroup"] != "no_photo"].copy()
df_photo["is_male"] = df_photo["subgroup"].isin(["black_man", "white_man"]).astype(int)

for pt in ["prompt_baseline", "prompt_ignore"]:
    pt_df = df_photo[df_photo["prompt_type"] == pt]

    r, p = pointbiserialr(pt_df["is_male"], pt_df["mentions_gender"])
    ct = pd.crosstab(pt_df["is_male"], pt_df["mentions_gender"])
    chi2, p_chi, _, _ = chi2_contingency(ct)

    print(f"--- {pt.upper()} ---")
    print(f"Point-biserial r = {r:.3f}, p = {p:.4f}")
    print(f"Chi-squared χ² = {chi2:.3f}, p = {p_chi:.4f}")
    print()

# Per-race breakdown
print("--- PROMPT_BASELINE: BREAKDOWN BY RACE ---")
for pt in ["prompt_baseline", "prompt_ignore"]:
    pt_df = df[df["prompt_type"] == pt]
    print(f"\n{pt.upper()}")
    for subgroup in ["black_man", "white_man", "black_woman", "white_woman"]:
        sub_df = pt_df[pt_df["subgroup"] == subgroup]
        if sub_df.empty:
            continue
        rate = sub_df["mentions_gender"].mean() * 100
        print(f"  {subgroup}: mentions gender {rate:.1f}% of runs")
