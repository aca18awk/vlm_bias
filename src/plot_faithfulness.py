import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

RESULTS_DIR = "../results"
EVAL_DIR = "../evaluations"
MAPPING_FILE = "../ground_truth_mapping.json"
EPSILON = 1e-5

# 1. Load Mapping
with open(MAPPING_FILE, "r") as f:
    mapping = json.load(f)
inv_map = {v: k for k, v in mapping.items()}

# 2. Extract Data (Combining Decisions and Evaluations)
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
        subgroup = f"{parts[1]}_{parts[2]}" if len(parts) == 3 else "no_image"

        run_dir = os.path.join(prompt_dir, encoded_name)
        eval_run_dir = os.path.join(EVAL_DIR, prompt_type, encoded_name)

        for run_idx in range(50):
            res_file = os.path.join(run_dir, f"response_n={run_idx}.json")
            eval_file = os.path.join(eval_run_dir, f"eval_n={run_idx}.json")

            if not os.path.exists(res_file) or not os.path.exists(eval_file):
                continue

            with open(res_file, "r") as f:
                res_data = json.load(f)
            with open(eval_file, "r") as f:
                eval_data = json.load(f)

            ans = res_data.get("parsed_answer", "")
            if (
                ans not in ["admit", "discharge"]
                or eval_data.get("mentions_race") is None
            ):
                continue

            rows.append(
                {
                    "prompt_type": prompt_type,
                    "vignette_id": vignette_id,
                    "subgroup": subgroup,
                    "decision_admit": 1 if ans == "admit" else 0,
                    # EIE: Did it mention race OR gender?
                    "mentions_demographic": 1
                    if (eval_data["mentions_race"] or eval_data["mentions_gender"])
                    else 0,
                }
            )

df = pd.DataFrame(rows)

# 3. Calculate EIE and CCE (KL Divergence) per Subgroup
metrics = []
for prompt_type in ["prompt_baseline", "prompt_ignore"]:
    prompt_df = df[df["prompt_type"] == prompt_type]

    for vignette_id in prompt_df["vignette_id"].unique():
        vig_df = prompt_df[prompt_df["vignette_id"] == vignette_id]

        baseline_df = vig_df[vig_df["subgroup"] == "no_image"]
        if baseline_df.empty:
            continue

        q_admit = max(baseline_df["decision_admit"].mean(), EPSILON)
        q_discharge = max(1 - q_admit, EPSILON)
        Q = np.array([q_admit, q_discharge])

        for subgroup in vig_df["subgroup"].unique():
            if subgroup == "no_image":
                continue

            sub_df = vig_df[vig_df["subgroup"] == subgroup]

            # Explanation-Implied Effect (EIE)
            p_demographic = sub_df["mentions_demographic"].mean()

            # Causal Concept Effect (CCE / KL Divergence)
            p_admit = max(sub_df["decision_admit"].mean(), EPSILON)
            p_discharge = max(1 - p_admit, EPSILON)
            P = np.array([p_admit, p_discharge])
            kl_div = stats.entropy(P, Q)

            metrics.append(
                {
                    "prompt_type": prompt_type,
                    "vignette_id": vignette_id,
                    "subgroup": subgroup,
                    "EIE": p_demographic,
                    "CCE": kl_div,
                }
            )

metrics_df = pd.DataFrame(metrics)

# 4. Standardize (Z-Score) to match the paper's axes
for prompt_type in ["prompt_baseline", "prompt_ignore"]:
    mask = metrics_df["prompt_type"] == prompt_type
    metrics_df.loc[mask, "EIE_Z"] = stats.zscore(metrics_df.loc[mask, "EIE"])
    metrics_df.loc[mask, "CCE_Z"] = stats.zscore(metrics_df.loc[mask, "CCE"])

# 5. Calculate Beta & Plot Faithfulness
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True, sharex=True)
fig.suptitle("Model Faithfulness: Causal Effect vs. Explanation Mentions", fontsize=16)

for idx, p_type in enumerate(["prompt_baseline", "prompt_ignore"]):
    data = metrics_df[metrics_df["prompt_type"] == p_type]
    ax = axes[idx]

    # Calculate Linear Regression (Beta)
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        data["CCE_Z"], data["EIE_Z"]
    )

    # Plot
    sns.regplot(
        data=data,
        x="CCE_Z",
        y="EIE_Z",
        ax=ax,
        scatter_kws={"alpha": 0.6, "s": 50},
        line_kws={"color": "red", "linewidth": 2},
    )

    title = (
        "Baseline (No Constraint)"
        if p_type == "prompt_baseline"
        else "Constraint (Ignore Photo)"
    )
    ax.set_title(f"{title}\nFaithfulness $\\beta$ = {slope:.3f} (p={p_value:.3f})")
    ax.set_xlabel(
        r"$\tilde{\mathbf{CE}}$ - Causal Concept Effect (KL Divergence Z-Score)"
    )
    if idx == 0:
        ax.set_ylabel(r"$\tilde{\mathbf{EE}}$ - Explanation Implied Effect (Z-Score)")
    else:
        ax.set_ylabel("")

plt.tight_layout()
plt.savefig("faithfulness_plot.png", dpi=300)
print("Saved plot to 'faithfulness_plot.png'!")
print("\n=== FAITHFULNESS SCORES (Beta) ===")
for p_type in ["prompt_baseline", "prompt_ignore"]:
    d = metrics_df[metrics_df["prompt_type"] == p_type]
    b, _, _, p, _ = stats.linregress(d["CCE_Z"], d["EIE_Z"])
    print(f"{p_type}: Beta = {b:.3f} (p-value: {p:.3f})")
