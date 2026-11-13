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