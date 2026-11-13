"""

NUMBERS FOR PARAGRAPH 4.2
Noise check for per-vignette effects.

A. Raw admit-rate table per (model, prompt, vignette) with Wilson CIs and flip flags.
B. Per-vignette effects with uncertainty:
     IPE    = grey - no_photo   Newcombe CI, permutation p
     CE_F   = face - no_photo   cluster bootstrap CI + p
     CE_FvG = face - grey       cluster bootstrap CI + p
   BH-FDR across the 36 cells, separately per model and effect type.
C. Noise floor for aggregates: mean |IPE| and range (max - min IPE) vs. permutation null.

Label permutations are drawn as hypergeometric counts (exactly equivalent to
shuffling the pooled grey/no_photo labels), and resampling runs within a face
is drawn as Binomial(n_face, p_face) (exactly equivalent to resampling runs
with replacement). This keeps everything vectorised.
"""
import os
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint, confint_proportions_2indep
from statsmodels.stats.multitest import multipletests

CONTROL_CSV = "Control_Aggregated_Data_updated.csv"
FACE_DATA = {
    # model: face run-level csv (only decision/photo columns are read)
    "llama":     "paper_results/Master_RunLevel_Data_EE.csv",
    "qwen":      "paper_results/Master_RunLevel_Data_EE.csv",
    "med_gemma": "paper_results/med_gemma/Master_RunLevel_Data_Medgemma.csv",
    # "gemma":   "paper_results/gemma/Master_RunLevel_Data_gemma.csv",
}
FACE_COLS = ["model", "prompt", "vignette_id", "vignette_class", "photo_id", "admit_decision"]
OUT_DIR = "paper_results/noise_check"

N_BOOT = 2000
N_PERM = 10000
ALPHA = 0.05
SEED = 42

rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def wilson(k, n):
    lo, hi = proportion_confint(k, n, alpha=ALPHA, method="wilson")
    return lo, hi


def newcombe_diff(k1, n1, k2, n2):
    """CI for p1 - p2 (Newcombe hybrid score, Wilson-based)."""
    lo, hi = confint_proportions_2indep(k1, n1, k2, n2, method="newcomb",
                                        compare="diff", alpha=ALPHA)
    return lo, hi


def fmt_rate(k, n):
    lo, hi = wilson(k, n)
    return f"{100*k/n:.1f} [{100*lo:.1f}, {100*hi:.1f}] ({n})"


def boot_face_rate(face_k, face_n, n_boot):
    """Cluster bootstrap of the face admit rate: resample faces, then runs within face."""
    idx = rng.integers(0, len(face_k), size=(n_boot, len(face_k)))
    n_star = face_n[idx]
    k_star = rng.binomial(n_star, face_k[idx] / face_n[idx])
    return k_star.sum(axis=1) / n_star.sum(axis=1)


def boot_control_rate(k, n, n_boot):
    return rng.binomial(n, k / n, size=n_boot) / n

# def boot_face_rate(face_k, face_n, n_boot):
#     """Cluster bootstrap of the face admit rate (Jeffreys-smoothed within face)."""
#     idx = rng.integers(0, len(face_k), size=(n_boot, len(face_k)))
#     n_star = face_n[idx]
#     p_star = rng.beta(face_k[idx] + 0.5, face_n[idx] - face_k[idx] + 0.5)
#     k_star = rng.binomial(n_star, p_star)
#     return k_star.sum(axis=1) / n_star.sum(axis=1)


# def boot_control_rate(k, n, n_boot):
#     """Jeffreys-smoothed so 0/n and n/n still carry realistic uncertainty."""
#     p = rng.beta(k + 0.5, n - k + 0.5, size=n_boot)
#     return rng.binomial(n, p) / n


def boot_summary(diffs, observed):
    lo, hi = np.percentile(diffs, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    # two-sided: twice the share of bootstrap differences on the other side of zero
    p = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    return observed, lo, hi, min(p, 1.0)


def perm_ipe_null(k_grey, n_grey, k_np, n_np, n_perm):
    """Null distribution of IPE under shuffling grey/no_photo labels."""
    K, N = k_grey + k_np, n_grey + n_np
    kg = rng.hypergeometric(K, N - K, n_grey, size=n_perm)
    return kg / n_grey - (K - kg) / n_np


def perm_p(null, observed):
    # add-one so p is never exactly 0 (needed for BH)
    return (np.sum(np.abs(null) >= abs(observed) - 1e-12) + 1) / (len(null) + 1)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_model(model):
    faces = pd.read_csv(FACE_DATA[model], usecols=FACE_COLS, low_memory=False)
    ctrl = pd.read_csv(CONTROL_CSV)
    faces = faces[faces.model == model].dropna(subset=["admit_decision"])
    ctrl = ctrl[ctrl.model == model]
    return faces, ctrl

# No-photo admit rate <=5% or >=95% (same rule as the regression)
EXCLUDE = {"qwen": [1, 4, 5, 10], "med_gemma": [4, 5, 10], "llama": []}

# ---------------------------------------------------------------------------
# A + B: per-cell table
# ---------------------------------------------------------------------------
def analyse_cells(model, faces, ctrl):
    rows = []
    for (prompt, vid), f in faces.groupby(["prompt", "vignette_id"]):
        if vid in EXCLUDE.get(model, []):
            continue
        c = ctrl[(ctrl.prompt == prompt) & (ctrl.vignette_id == vid)]
        npd = c[c.condition == "no_photo"].iloc[0]
        gry = c[c.condition == "grey_rect"].iloc[0]
        k_np, n_np = int(npd.n_admits), int(npd.n_total)
        k_g, n_g = int(gry.n_admits), int(gry.n_total)

        per_face = f.groupby("photo_id").admit_decision.agg(["sum", "count"])
        face_k = per_face["sum"].to_numpy()
        face_n = per_face["count"].to_numpy()
        k_f, n_f = int(face_k.sum()), int(face_n.sum())

        p_np, p_g, p_f = k_np / n_np, k_g / n_g, k_f / n_f

        # IPE: Newcombe CI + permutation p
        ipe = p_g - p_np
        ipe_lo, ipe_hi = newcombe_diff(k_g, n_g, k_np, n_np)
        ipe_p = perm_p(perm_ipe_null(k_g, n_g, k_np, n_np, N_PERM), ipe)

        # CE_F and CE_FvG: cluster bootstrap (shared face draws)
        face_star = boot_face_rate(face_k, face_n, N_BOOT)
        np_star = boot_control_rate(k_np, n_np, N_BOOT)
        g_star = boot_control_rate(k_g, n_g, N_BOOT)
        cef = boot_summary(face_star - np_star, p_f - p_np)
        cefg = boot_summary(face_star - g_star, p_f - p_g)

        flip_grey = (p_g > 0.5) != (p_np > 0.5)
        flip_face = (p_f > 0.5) != (p_np > 0.5)

        rows.append({
            "model": model, "prompt": prompt, "vignette_id": vid,
            "vignette_class": f.vignette_class.iloc[0],
            "k_no_photo": k_np, "n_no_photo": n_np,
            "k_grey": k_g, "n_grey": n_g,
            "k_face": k_f, "n_face": n_f, "n_faces": len(face_k),
            "No-photo %": fmt_rate(k_np, n_np),
            "Grey %": fmt_rate(k_g, n_g),
            "Face %": fmt_rate(k_f, n_f),
            "flip_grey": flip_grey, "flip_face": flip_face,
            "IPE": ipe, "IPE_lo": ipe_lo, "IPE_hi": ipe_hi, "IPE_p": ipe_p,
            "CE_F": cef[0], "CE_F_lo": cef[1], "CE_F_hi": cef[2], "CE_F_p": cef[3],
            "CE_FvG": cefg[0], "CE_FvG_lo": cefg[1], "CE_FvG_hi": cefg[2], "CE_FvG_p": cefg[3],
        })
    cells = pd.DataFrame(rows)

    # BH-FDR across the 36 cells, per effect type (this function is per model)
    for eff in ["IPE", "CE_F", "CE_FvG"]:
        rej, q, _, _ = multipletests(cells[f"{eff}_p"], alpha=ALPHA, method="fdr_bh")
        cells[f"{eff}_q"] = q
        cells[f"{eff}_sig"] = rej

    # A flip only counts if the corresponding effect is also significant after FDR
    cells["flip_grey_sig"] = cells.flip_grey & cells.IPE_sig
    cells["flip_face_sig"] = cells.flip_face & cells.CE_F_sig
    cells["Flip?"] = [
        "+".join(name for name, on in [("grey", g), ("face", fc)] if on) + ("*" if (gs or fs) else "")
        for g, fc, gs, fs in zip(cells.flip_grey, cells.flip_face,
                                 cells.flip_grey_sig, cells.flip_face_sig)
    ]
    return cells


# ---------------------------------------------------------------------------
# C: noise floor for aggregates (per model, prompt)
# ---------------------------------------------------------------------------
def noise_floor(model, cells):
    rows = []
    for prompt, cp in cells.groupby("prompt"):
        # shuffle labels within every vignette, same permutation index across vignettes
        null_ipe = np.stack([
            perm_ipe_null(r.k_grey, r.n_grey, r.k_no_photo, r.n_no_photo, N_PERM)
            for r in cp.itertuples()
        ], axis=1)  # (N_PERM, n_vignettes)

        obs_mean_abs = cp.IPE.abs().mean()
        obs_range = cp.IPE.max() - cp.IPE.min()
        null_mean_abs = np.abs(null_ipe).mean(axis=1)
        null_range = null_ipe.max(axis=1) - null_ipe.min(axis=1)

        rows.append({
            "model": model, "prompt": prompt, "n_vignettes": len(cp),
            "mean_abs_IPE": obs_mean_abs,
            "mean_abs_IPE_null_mean": null_mean_abs.mean(),
            "mean_abs_IPE_null_95": np.percentile(null_mean_abs, 95),
            "mean_abs_IPE_p": (np.sum(null_mean_abs >= obs_mean_abs) + 1) / (N_PERM + 1),
            "range_IPE": obs_range,
            "range_IPE_null_mean": null_range.mean(),
            "range_IPE_null_95": np.percentile(null_range, 95),
            "range_IPE_p": (np.sum(null_range >= obs_range) + 1) / (N_PERM + 1),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    all_cells, all_floor = [], []
    for model in FACE_DATA:
        print(f"Processing {model}...")
        faces, ctrl = load_model(model)
        cells = analyse_cells(model, faces, ctrl)
        all_cells.append(cells)
        all_floor.append(noise_floor(model, cells))

    cells = pd.concat(all_cells, ignore_index=True)
    floor = pd.concat(all_floor, ignore_index=True)

    # A: full raw admit-rate table (appendix)
    table_a = cells[["model", "prompt", "vignette_id", "vignette_class",
                     "No-photo %", "Grey %", "Face %", "Flip?"]]
    table_a.to_csv(f"{OUT_DIR}/A_admit_rates_full.csv", index=False)

    # A (main text): only vignettes with at least one significant effect
    any_sig = cells.IPE_sig | cells.CE_F_sig | cells.CE_FvG_sig
    table_a[any_sig].to_csv(f"{OUT_DIR}/A_admit_rates_significant.csv", index=False)

    # B: per-cell effects with CIs, raw p and BH q
    cells.to_csv(f"{OUT_DIR}/B_vignette_effects.csv", index=False)

    # B summary: "k of 36 cells significant after FDR"
    summary = []
    for model, cm in cells.groupby("model", sort=False):
        for eff in ["IPE", "CE_F", "CE_FvG"]:
            summary.append({
                "model": model, "effect": eff,
                "k_sig": int(cm[f"{eff}_sig"].sum()), "n_cells": len(cm),
                "k_sig_pos": int((cm[f"{eff}_sig"] & (cm[eff] > 0)).sum()),
                "k_sig_neg": int((cm[f"{eff}_sig"] & (cm[eff] < 0)).sum()),
                "max_abs_sig_effect": cm.loc[cm[f"{eff}_sig"], eff].abs().max(),
            })
    summary = pd.DataFrame(summary)
    summary.to_csv(f"{OUT_DIR}/B_fdr_summary.csv", index=False)

    # C: noise floor
    floor.to_csv(f"{OUT_DIR}/C_noise_floor.csv", index=False)

    # Console report
    print("\n=== B. Cells significant after BH-FDR ===")
    for r in summary.itertuples():
        print(f"{r.model:10s} {r.effect:7s} {r.k_sig:2d} of {r.n_cells} cells "
              f"(+{r.k_sig_pos} / -{r.k_sig_neg})")

    print("\n=== C. Noise floor (IPE) ===")
    for r in floor.itertuples():
        print(f"{r.model:10s} {r.prompt:12s} mean|IPE| = {r.mean_abs_IPE:.3f} vs floor "
              f"{r.mean_abs_IPE_null_mean:.3f} (p = {r.mean_abs_IPE_p:.4f});  "
              f"range = {r.range_IPE:.3f} vs floor {r.range_IPE_null_mean:.3f} "
              f"(p = {r.range_IPE_p:.4f})")

    print(f"\nSaved tables to {OUT_DIR}/")





"""
Pooled (across-vignette) photo effects per (model, prompt).

Estimand: the mean over vignettes of (rate_A - rate_B), i.e. each vignette counts equally.
  CE_F   = face      - no_photo
  CE_FvG = face      - grey_rect
  IPE    = grey_rect - no_photo

Uncertainty: bootstrap that resamples the 50 faces (with all their runs across
vignettes, since the same faces appear in every vignette) and resamples control
runs within each vignette. Vignettes are treated as fixed (the estimand is for these
vignettes). BH-FDR is applied across the (model, prompt) cells of each contrast.
"""
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

CONTROL_CSV = "Control_Aggregated_Data_updated.csv"
FACE_DATA = {
    "llama":     "paper_results/Master_RunLevel_Data_EE.csv",
    "qwen":      "paper_results/Master_RunLevel_Data_EE.csv",
    "med_gemma": "paper_results/med_gemma/Master_RunLevel_Data_Medgemma.csv",
}
FACE_COLS = ["model", "prompt", "vignette_id", "photo_id", "admit_decision"]

# Set True to apply the floor/ceiling rule (same as regression / noise check)
EXCLUDE_FLOOR_CEILING = False
EXCLUDE = {"qwen": [1, 4, 5, 10], "med_gemma": [4, 5, 10], "llama": []}

N_BOOT = 2000
ALPHA = 0.05
rng = np.random.default_rng(42)


def pooled_effects(faces, ctrl, model, prompt):
    f = faces[(faces.model == model) & (faces.prompt == prompt)]
    c = ctrl[(ctrl.model == model) & (ctrl.prompt == prompt)]
    vids = sorted(f.vignette_id.unique())
    if EXCLUDE_FLOOR_CEILING:
        vids = [v for v in vids if v not in EXCLUDE.get(model, [])]
    f = f[f.vignette_id.isin(vids)]

    # face counts: faces x vignettes
    k = f.pivot_table(index="photo_id", columns="vignette_id",
                      values="admit_decision", aggfunc="sum").reindex(columns=vids)
    n = f.pivot_table(index="photo_id", columns="vignette_id",
                      values="admit_decision", aggfunc="count").reindex(columns=vids)
    k, n = k.fillna(0).to_numpy(), n.fillna(0).to_numpy()

    def ctrl_counts(cond):
        cc = c[c.condition == cond].set_index("vignette_id").reindex(vids)
        return cc.n_admits.to_numpy().astype(int), cc.n_total.to_numpy().astype(int)
    k_np, n_np = ctrl_counts("no_photo")
    k_g, n_g = ctrl_counts("grey_rect")

    # observed (each vignette weighted equally)
    face_v = k.sum(0) / n.sum(0)
    np_v, g_v = k_np / n_np, k_g / n_g
    obs = {"CE_F": np.mean(face_v - np_v),
           "CE_FvG": np.mean(face_v - g_v),
           "IPE": np.mean(g_v - np_v)}

    # bootstrap
    F = k.shape[0]
    p_face = np.divide(k, n, out=np.zeros_like(k, dtype=float), where=n > 0)
    idx = rng.integers(0, F, size=(N_BOOT, F))                 # resample faces
    n_star = n[idx]                                            # B x F x V
    k_star = rng.binomial(n_star.astype(int), p_face[idx])     # resample runs within face
    face_v_star = k_star.sum(1) / n_star.sum(1)                # B x V
    np_v_star = rng.binomial(n_np, np_v, size=(N_BOOT, len(vids))) / n_np
    g_v_star = rng.binomial(n_g, g_v, size=(N_BOOT, len(vids))) / n_g

    boots = {"CE_F": (face_v_star - np_v_star).mean(1),
             "CE_FvG": (face_v_star - g_v_star).mean(1),
             "IPE": (g_v_star - np_v_star).mean(1)}

    row = {"model": model, "prompt": prompt, "n_vignettes": len(vids)}
    for eff, d in boots.items():
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = 2 * min(np.mean(d <= 0), np.mean(d >= 0))
        row.update({eff: obs[eff], f"{eff}_lo": lo, f"{eff}_hi": hi,
                    f"{eff}_p": max(p, 1 / N_BOOT)})
    return row


if __name__ == "__main__":
    ctrl = pd.read_csv(CONTROL_CSV)
    rows = []
    for model, path in FACE_DATA.items():
        faces = pd.read_csv(path, usecols=FACE_COLS, low_memory=False)
        faces = faces[faces.model == model].dropna(subset=["admit_decision"])
        for prompt in ["baseline", "acknowledge", "ignore"]:
            rows.append(pooled_effects(faces, ctrl, model, prompt))
    res = pd.DataFrame(rows)

    for eff in ["CE_F", "CE_FvG", "IPE"]:
        rej, q, _, _ = multipletests(res[f"{eff}_p"], alpha=ALPHA, method="fdr_bh")
        res[f"{eff}_q"], res[f"{eff}_sig"] = q, rej

    res.to_csv("paper_results/noise_check/pooled_effects.csv", index=False)

    for eff in ["CE_F", "CE_FvG", "IPE"]:
        print(f"\n=== {eff} (pooled across vignettes, pp) ===")
        for r in res.itertuples():
            v, lo, hi = 100 * getattr(r, eff), 100 * getattr(r, f"{eff}_lo"), 100 * getattr(r, f"{eff}_hi")
            q, sig = getattr(r, f"{eff}_q"), getattr(r, f"{eff}_sig")
            print(f"{r.model:10s} {r.prompt:12s} {v:+5.1f}  [{lo:+5.1f}, {hi:+5.1f}]  "
                  f"q = {q:.3f} {'*' if sig else ''}")