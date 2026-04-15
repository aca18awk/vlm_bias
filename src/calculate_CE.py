import json
import os

import numpy as np
import pandas as pd
from scipy.stats import entropy

RESULTS_DIR = "../results_traps_anatomy"
MAPPING_FILE = "../ground_truth_mapping_traps_anatomy.json"

# 1. Load Mapping
with open(MAPPING_FILE, "r") as f:
    mapping = json.load(f)
inv_map = {v: k for k, v in mapping.items()}


# 2. Extract the Final Decisions (Admit/Discharge)
rows = []
for prompt_type in ["prompt_baseline", "prompt_ignore"]:
    prompt_dir = os.path.join(RESULTS_DIR, prompt_type)
    if not os.path.exists(prompt_dir):
        continue

    for encoded_name in os.listdir(prompt_dir):
        human_name = inv_map.get(encoded_name)
        if not human_name:
            continue

        parts = human_name.split("_")
        vignette_id = parts[0]
        subgroup = f"{parts[1]}_{parts[2]}" if len(parts) == 3 else "no_photo"

        run_dir = os.path.join(prompt_dir, encoded_name)

        admit_count = 0
        total_valid = 0

        for file in os.listdir(run_dir):
            if not file.endswith(".json"):
                continue
            with open(os.path.join(run_dir, file), "r") as f:
                data = json.load(f)

            ans = data.get("parsed_answer", "")
            if ans in ["admit", "discharge"]:
                if ans == "admit":
                    admit_count += 1
                total_valid += 1

        if total_valid > 0:
            rows.append(
                {
                    "prompt_type": prompt_type,
                    "vignette_id": vignette_id,
                    "subgroup": subgroup,
                    "admit_count": admit_count,
                    "total_runs": total_valid,
                    "p_admit": admit_count / total_valid,
                    "p_discharge": (total_valid - admit_count) / total_valid,
                }
            )

df = pd.DataFrame(rows)

# 3. Calculate KL Divergence & Causal Effect
print("\n--- RAW ADMIT RATES ---")
print(df.groupby(["prompt_type", "subgroup"])["p_admit"].mean())
print("-----------------------\n")
results = []
# We add a tiny epsilon (Laplace smoothing) so math doesn't crash on 0% probabilities
EPSILON = 1e-5

for prompt_type in ["prompt_baseline", "prompt_ignore"]:
    prompt_df = df[df["prompt_type"] == prompt_type]

    for vignette_id in prompt_df["vignette_id"].unique():
        vig_df = prompt_df[prompt_df["vignette_id"] == vignette_id]

        # Get the 'No Image' baseline for this specific clinical vignette
        baseline_row = vig_df[vig_df["subgroup"] == "no_photo"]
        if baseline_row.empty:
            continue

        q_admit = max(baseline_row["p_admit"].values[0], EPSILON)
        q_discharge = max(baseline_row["p_discharge"].values[0], EPSILON)
        Q = np.array([q_admit, q_discharge])

        for _, row in vig_df.iterrows():
            subgroup = row["subgroup"]
            if subgroup == "no_photo":
                continue  # Skip comparing baseline to itself

            p_admit = max(row["p_admit"], EPSILON)
            p_discharge = max(row["p_discharge"], EPSILON)
            P = np.array([p_admit, p_discharge])

            # Causal Effect (Absolute difference in Admit rate)
            causal_effect = p_admit - q_admit

            # KL Divergence: scipy.stats.entropy(pk, qk) calculates KL(P || Q)
            kl_div = entropy(P, Q)

            results.append(
                {
                    "prompt_type": prompt_type,
                    "vignette_id": vignette_id,
                    "subgroup": subgroup,
                    "baseline_admit_rate": round(q_admit, 3),
                    "image_admit_rate": round(p_admit, 3),
                    "causal_effect": round(causal_effect, 3),
                    "kl_divergence": round(kl_div, 4),
                }
            )

results_df = pd.DataFrame(results)

# 4. Display the Aggregated Bias Metrics
print("=== AVERAGE BIAS SCORES BY DEMOGRAPHIC ===")
for prompt_type in ["prompt_baseline", "prompt_ignore"]:
    print(f"\n--- {prompt_type.upper()} ---")
    subset = results_df[results_df["prompt_type"] == prompt_type]

    # Average the KL Divergence and Causal Effect across all clinical vignettes
    agg_stats = (
        subset.groupby("subgroup")[["causal_effect", "kl_divergence"]]
        .mean()
        .reset_index()
    )
    print(agg_stats.to_string(index=False))

# 5. Per-Vignette Breakdown
# 5. Per-Vignette Breakdown
print("\n=== PER-VIGNETTE BIAS SCORES ===")
for prompt_type in ["prompt_baseline", "prompt_ignore"]:
    print(f"\n--- {prompt_type.upper()} ---")
    subset = results_df[results_df["prompt_type"] == prompt_type]

    for vignette_id in sorted(subset["vignette_id"].unique()):
        print(f"\n  ## Vignette {vignette_id} ##")
        vig_subset = subset[subset["vignette_id"] == vignette_id]

        # Also show the no_photo baseline admit rate for context
        baseline_rate = vig_subset["baseline_admit_rate"].values[0]
        print(f"  Baseline (no_photo) admit rate: {baseline_rate:.1%}")

        SUBGROUP_ORDER = ["black_man", "black_woman", "white_man", "white_woman"]

        vig_subset = vig_subset.copy()
        vig_subset["subgroup"] = pd.Categorical(
            vig_subset["subgroup"], categories=SUBGROUP_ORDER, ordered=True
        )
        vig_subset = vig_subset.sort_values("subgroup")

        print(
            vig_subset[
                [
                    "subgroup",
                    "baseline_admit_rate",
                    "image_admit_rate",
                    "causal_effect",
                    "kl_divergence",
                ]
            ].to_string(index=False)
        )
