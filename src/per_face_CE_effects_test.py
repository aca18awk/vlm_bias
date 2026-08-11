"""
Per-subgroup CE_FvG for 6 focal cells.

CE_FvG (subgroup) = mean admit rate (subgroup faces) − mean admit rate (grey rectangle)

Cells:
  Qwen:  V11/baseline (+0.26), V03/acknowledge (+0.23), V09/ignore (+0.21)
  Llama: V09/baseline (+0.27), V11/acknowledge (+0.30), V07/ignore (+0.31)
"""

import json
import os

BASE = "../15_Apr/vignettes_nature_paper"

MAPPING_CFD     = os.path.join(BASE, "ground_truth_mapping.json")
MAPPING_CONTROL = os.path.join(BASE, "ground_truth_mapping_control.json")

MODEL_SUFFIX = {"qwen": "qwen25", "llama": "llama32"}
PROMPT_DIR   = {"baseline": "prompt_baseline", "acknowledge": "prompt_acknowledge", "ignore": "prompt_ignore"}
SUBGROUPS    = ["AF", "AM", "BF", "BM", "IF", "IM", "LF", "LM", "WF", "WM"]

CELLS = [
    ("qwen",  "V11", "baseline",   +0.26),
    ("qwen",  "V03", "acknowledge", +0.23),
    ("qwen",  "V09", "ignore",      +0.21),
    ("llama", "V09", "baseline",    +0.27),
    ("llama", "V11", "acknowledge", +0.30),
    ("llama", "V07", "ignore",      +0.31),
]

with open(MAPPING_CFD)     as f: cfd_map     = json.load(f)
with open(MAPPING_CONTROL) as f: control_map = json.load(f)


def parse_decision(path):
    with open(path) as f:
        d = json.load(f)
    ans = d.get("parsed_answer", "").strip().lower()
    if ans == "admit":
        return 1
    if ans == "discharge":
        return 0
    return None  # ERROR_PARSING or missing


def read_responses(folder):
    if not os.path.isdir(folder):
        return []
    results = []
    for fname in os.listdir(folder):
        if fname.endswith(".json"):
            v = parse_decision(os.path.join(folder, fname))
            results.append(v)
    return results


def admit_rate(responses):
    valid = [r for r in responses if r is not None]
    errors = len(responses) - len(valid)
    if not valid:
        return None, 0, errors
    return sum(valid) / len(valid), len(valid), errors


def run_cell(model, vignette, prompt, expected_CE_FvG):
    vv = vignette.lstrip("V").zfill(2)          # "V11" -> "11"
    suffix    = MODEL_SUFFIX[model]
    pdir      = PROMPT_DIR[prompt]
    cfd_dir   = os.path.join(BASE, f"results_{suffix}_cfd",     pdir)
    ctrl_dir  = os.path.join(BASE, f"results_{suffix}_control", pdir)

    # ── Grey rectangle baseline ───────────────────────────────────────────────
    grey_key     = f"{vv}_grey_rect"
    grey_encoded = control_map.get(grey_key)
    if grey_encoded is None:
        print(f"  [WARN] No grey_rect entry for {grey_key}")
        return

    grey_responses = read_responses(os.path.join(ctrl_dir, grey_encoded))
    grey_rate, grey_n, grey_err = admit_rate(grey_responses)
    if grey_rate is None:
        print(f"  [WARN] No grey responses found in {ctrl_dir}/{grey_encoded}")
        return

    # ── Per-face (individual folder) ──────────────────────────────────────────
    rows = []
    for sg in SUBGROUPS:
        for idx in range(1, 6):
            key = f"{vv}_{sg}_{idx}"
            enc = cfd_map.get(key)
            if enc is None:
                continue
            folder = os.path.join(cfd_dir, enc)
            responses = read_responses(folder)
            rate, n, err = admit_rate(responses)
            ce = (rate - grey_rate) if rate is not None else None
            rows.append({
                "face":      key,
                "encoded":   enc,
                "subgroup":  sg,
                "n_valid":   n,
                "n_error":   err,
                "face_rate": rate,
                "grey_rate": grey_rate,
                "CE_FvG":    ce,
                "abs_CE":    abs(ce) if ce is not None else -1,
                "folder":    folder,
            })

    rows.sort(key=lambda r: r["abs_CE"], reverse=True)

    # ── Sanity check: weighted mean CE_FvG vs expected ────────────────────────
    valid_rows    = [r for r in rows if r["CE_FvG"] is not None]
    weighted_mean = sum(r["CE_FvG"] * r["n_valid"] for r in valid_rows) / \
                    sum(r["n_valid"] for r in valid_rows) if valid_rows else None

    # ── Print ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"  {model.upper()}  |  {vignette}  |  {prompt}")
    print(f"  Grey admit rate: {grey_rate:.3f}  (n={grey_n}, errors={grey_err})")
    print(f"  Expected CE_FvG from paper: {expected_CE_FvG:+.2f}")
    if weighted_mean is not None:
        match = abs(weighted_mean - expected_CE_FvG) < 0.05
        print(f"  Weighted-mean CE_FvG across all faces: {weighted_mean:+.3f}  "
              f"{'✓ matches' if match else '⚠ MISMATCH — check filtering'}")
    print(f"{'='*75}")
    print(f"  {'Face':<14} {'SG':>4} {'n':>5} {'err':>5} {'Face rate':>10} {'CE_FvG':>8}")
    print(f"  {'-'*50}")
    for r in rows:
        rate_str = f"{r['face_rate']:.3f}" if r["face_rate"] is not None else "  N/A"
        ce_str   = f"{r['CE_FvG']:+.3f}"  if r["CE_FvG"]   is not None else "  N/A"
        print(f"  {r['face']:<14} {r['subgroup']:>4} {r['n_valid']:>5} {r['n_error']:>5} "
              f"{rate_str:>10} {ce_str:>8}")

    print(f"\n  Top faces by |CE_FvG|:")
    for r in rows[:5]:
        ce_str = f"{r['CE_FvG']:+.3f}" if r["CE_FvG"] is not None else "N/A"
        print(f"    {r['face']:<14} (encoded: {r['encoded']})  CE_FvG={ce_str}")
        print(f"      {r['folder']}/")

    return rows


for model, vignette, prompt, expected in CELLS:
    run_cell(model, vignette, prompt, expected)
