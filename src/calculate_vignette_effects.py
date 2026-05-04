"""
Per-vignette effect table: IPE, CE_F, CE_FvG  ×  prompt  ×  model.

  IPE    = grey_rect − no_photo        image-presence effect
  CE_F   = mean(faces) − no_photo      combined face effect
  CE_FvG = mean(faces) − grey_rect     face-content effect

Output per model:
  • vignette_effects_<model>.csv   (wide, for LaTeX)
  • printed table to stdout
"""
import os

import numpy as np
import pandas as pd

CSV_PATH = "Master_Aggregated_Data.csv"
CONTROL_CSV_PATH = "Control_Aggregated_Data.csv"
OUTPUT_DIR = "."

# Optional: fill in to get human-readable domain labels in the table.
# Keys are vignette_id integers as they appear in the CSV.
DOMAIN_MAP = {
    1: "TIA",
    2: "Headache",
    3: "Focal neuro",
    4: "Seizure",
    5: "Thunderclap",
    6: "DVT",
    7: "Appendicitis",
    8: "Pyelonephritis",
    9: "Septic arth.",
    10: "DKA",
    11: "Suicidal id.",
    12: "Psychosis",
}

PROMPT_ORDER = ["baseline", "ignore", "acknowledge"]
EFFECTS = ["IPE", "CE_F", "CE_FvG"]
EPSILON = 1e-5


def _binom_var(p, n):
    return p * (1 - p) / np.maximum(n, 1)


# ── Load data ──────────────────────────────────────────────────────────────────
df_raw = pd.read_csv(CSV_PATH)
ctrl_raw = pd.read_csv(CONTROL_CSV_PATH)

# Control baselines: one row per (model, prompt, vignette_id, condition)
ctrl_pivot = ctrl_raw.pivot_table(
    index=["model", "prompt", "vignette_id"],
    columns="condition",
    values=["admit_rate", "n_total"],
    aggfunc="first",
)
ctrl_pivot.columns = [f"{cond}_{stat}" for stat, cond in ctrl_pivot.columns]
ctrl_pivot = ctrl_pivot.reset_index()

# Vignette class (borderline / clear)
vig_class = (
    df_raw[["vignette_id", "vignette_class"]]
    .drop_duplicates("vignette_id")
    .set_index("vignette_id")["vignette_class"]
)

# ── Per-vignette face mean and sampling variance of that mean ──────────────────
def _face_stats(grp):
    p_i = grp["admit_rate"].values
    n_i = grp["n_total"].values
    k = len(grp)
    p_bar = p_i.mean()
    # Var(p̄) = (1/k²) Σ p_i(1−p_i)/N_i
    var_p_bar = (_binom_var(p_i, n_i)).sum() / (k ** 2)
    return pd.Series({"p_face": p_bar, "var_p_face": var_p_bar, "n_faces": k})


face_df = (
    df_raw.groupby(["model", "prompt", "vignette_id"])
    .apply(_face_stats)
    .reset_index()
)

merged = face_df.merge(ctrl_pivot, on=["model", "prompt", "vignette_id"], how="left")

# Clip baselines to avoid degenerate SE
for cond in ["no_photo", "grey_rect"]:
    merged[f"{cond}_admit_rate"] = merged[f"{cond}_admit_rate"].clip(EPSILON, 1 - EPSILON)

# ── Compute effects ────────────────────────────────────────────────────────────
nop = merged["no_photo_admit_rate"]
gry = merged["grey_rect_admit_rate"]
N_nop = merged["no_photo_n_total"].fillna(1)
N_gry = merged["grey_rect_n_total"].fillna(1)
vpf = merged["var_p_face"]
pf = merged["p_face"]

merged["IPE"]       = (gry - nop).round(4)
merged["IPE_se"]    = np.sqrt(_binom_var(gry, N_gry) + _binom_var(nop, N_nop)).round(4)

merged["CE_F"]      = (pf - nop).round(4)
merged["CE_F_se"]   = np.sqrt(vpf + _binom_var(nop, N_nop)).round(4)

merged["CE_FvG"]    = (pf - gry).round(4)
merged["CE_FvG_se"] = np.sqrt(vpf + _binom_var(gry, N_gry)).round(4)

# ── Build wide output per model ────────────────────────────────────────────────
for model_name in sorted(merged["model"].unique()):
    mdf = merged[merged["model"] == model_name]
    vignette_ids = sorted(mdf["vignette_id"].unique())

    records = []
    for vid in vignette_ids:
        row: dict = {
            "vignette_id": vid,
            "domain": DOMAIN_MAP.get(vid, str(vid)),
            "vignette_class": vig_class.get(vid, ""),
        }
        vdf = mdf[mdf["vignette_id"] == vid]
        for prompt in PROMPT_ORDER:
            prow = vdf[vdf["prompt"] == prompt]
            for eff in EFFECTS:
                if prow.empty:
                    row[f"{prompt}_{eff}"] = None
                    row[f"{prompt}_{eff}_se"] = None
                else:
                    r = prow.iloc[0]
                    row[f"{prompt}_{eff}"] = r[eff]
                    row[f"{prompt}_{eff}_se"] = r[f"{eff}_se"]
        records.append(row)

    out_df = pd.DataFrame(records)

    out_csv = os.path.join(OUTPUT_DIR, f"vignette_effects_{model_name}.csv")
    out_df.to_csv(out_csv, index=False, float_format="%.4f")

    # ── Pretty-print ──────────────────────────────────────────────────────────
    W = 7  # column width for each effect value
    GAP = "  "

    # Header row 1: prompt labels spanning 3 effect columns each
    span = W * 3 + 4  # 3 values + 2 separators between them
    h1 = f"{'V':>3}  {'Domain':<12}  {'Class':<10}"
    for prompt in PROMPT_ORDER:
        h1 += GAP + f"{prompt.upper():^{span}}"

    # Header row 2: effect names
    h2 = f"{'':>3}  {'':12}  {'':10}"
    for _ in PROMPT_ORDER:
        for eff in EFFECTS:
            h2 += GAP + f"{eff:>{W}}"

    sep = "─" * len(h1)

    print(f"\n{'═' * len(h1)}")
    print(f"  Model: {model_name}")
    print(f"{'═' * len(h1)}")
    print(h1)
    print(h2)
    print(sep)

    for _, row in out_df.iterrows():
        line = f"{str(row['vignette_id']):>3}  {str(row['domain']):<12}  {str(row['vignette_class']):<10}"
        for prompt in PROMPT_ORDER:
            for eff in EFFECTS:
                val = row.get(f"{prompt}_{eff}")
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    cell = f"{'—':>{W}}"
                else:
                    cell = f"{float(val):>+{W}.3f}"
                line += GAP + cell
        print(line)

    print(f"\nSaved → {out_csv}")
