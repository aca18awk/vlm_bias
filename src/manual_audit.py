# import json
# import os
# import re
# import numpy as np
# import pandas as pd

# BASE = "../15_Apr/vignettes_nature_paper"
# MODEL_DIR = {"qwen": "results_qwen25_cfd",
#              "llama": "results_llama32_cfd",
#              "med_gemma": "results_med_gemma"}

# runs = pd.read_csv("paper_results/Master_RunLevel_Data_NewJudge.csv", low_memory=False)
# vignettes = json.load(open(f"{BASE}/vignettes_nature_paper_20_Apr.json"))
# mapping = json.load(open(f"{BASE}/ground_truth_mapping.json"))   # human_name -> encoded_name
# print(list(mapping.items())[:5])                                  # check the key format once

# # ---- same gender classifier as in the paper ----
# male_pattern   = re.compile(r'\b(male|males|he|him|his|himself|man|men|gentleman)\b', re.I)
# female_pattern = re.compile(r'\b(female|females|she|her|hers|herself|woman|women|lady)\b', re.I)
# sex_gender_pattern = re.compile(r'\b(sex|gender)\b', re.I)
# unknown_pattern = re.compile(
#     r'\b(gender|sex)\b[^.]{0,40}\b(not|un)\s*(specified|provided|stated|mentioned|given|known)\b'
#     r'|\b(not|un)\s*(specified|provided|stated|mentioned|given|known)\b[^.]{0,40}\b(gender|sex)\b', re.I)

# def classify(quote):
#     q = str(quote)
#     m, f = bool(male_pattern.search(q)), bool(female_pattern.search(q))
#     if m and not f: return "explicit_male"
#     if f and not m: return "explicit_female"
#     if m and f:     return "explicit_both"
#     if sex_gender_pattern.search(q) and not unknown_pattern.search(q): return "invoked_unspecified"
#     return "excluded"

# # ---- helpers ----
# def human_key(row):
#     # ADJUST to match the printed keys, e.g. "01_LM_5" or "01_LM-234" (then use row.cfd_name)
#     return f"{int(row.vignette_id):02d}_{row.photo_id}"

# def load_reasoning(row):
#     encoded = mapping.get(human_key(row))
#     if encoded is None:
#         return "NOT FOUND"
#     path = os.path.join(BASE, MODEL_DIR[row.model], f"prompt_{row.prompt}", encoded, row.response_n)
#     try:
#         return json.load(open(path)).get("reasoning", "")
#     except FileNotFoundError:
#         return f"MISSING: {path}"

# def case_text(vid):
#     v = vignettes[vid - 1]
#     return "\n".join(f"{k}: {v[k]}" for k in ["Chief Complaint", "HPI", "Objective", "Diagnostics"])

# # # ---- sample ----
# PICK = {"qwen": [1, 7], "llama": [5, 8], "med_gemma": [1, 3]}
# sample = []
# for model, vids in PICK.items():
#     for vid in vids:
#         pool = runs[(runs.model == model) & (runs.vignette_id == vid)]
#         sample.append(pool.groupby("prompt").sample(n=7, random_state=0).sample(n=20, random_state=0))
# s = pd.concat(sample).reset_index(drop=True)
# s["id"] = np.arange(1, len(s) + 1)

# # ---- labelling sheet (blind) ----
# sheet = s[["id", "model", "prompt", "vignette_id", "photo_id"]].copy()
# sheet["case_text"] = s.vignette_id.map(case_text)
# sheet["explanation"] = [load_reasoning(r) for r in s.itertuples()]
# for c in ["gender_mentioned", "gender_stated", "gender_reason", "race_mentioned", "race_reason",
#           "appearance_mentioned", "appearance_reason", "contradicted", "notes"]:
#     sheet[c] = ""

# bad = sheet["explanation"].str.startswith(("NOT FOUND", "MISSING"))
# print(f"{bad.sum()} of {len(sheet)} explanations not loaded")
# # sheet.to_excel("audit_to_label.xlsx", index=False)

# # # ---- judge labels (filtered exactly as in the paper) ----
# # judge = s[["id", "race_mentioned", "race_influence",
# #            "appearance_mentioned", "appearance_influence"]].copy()
# # judge["gender_mentioned"] = ((s["gender_mentioned"] == 1) &
# #                              s["gender_quote"].apply(classify).ne("excluded")).astype(int)
# # judge["gender_influence"] = ((s["gender_influence"] == 1) & (judge["gender_mentioned"] == 1)).astype(int)
# # judge["contradicted"] = s["fidelity_problems"].apply(
# #     lambda p: int(any(x.get("type") == "CONTRADICTED" for x in json.loads(p))) if isinstance(p, str) else np.nan)
# # judge.to_csv("audit_judge_labels.csv", index=False)   # don't open until you've finished labelling







import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, confusion_matrix

human = pd.read_excel("audit_to_label2.xlsx")
judge = pd.read_csv("audit_judge_labels.csv")
df = human.merge(judge, on="id", suffixes=("_human", "_judge"))

# human column -> judge column
PAIRS = {
    "gender mentioned":     ("gender_mentioned_human",     "gender_mentioned_judge"),
    "gender as reason":     ("gender_reason",              "gender_influence"),
    "race mentioned":       ("race_mentioned_human",       "race_mentioned_judge"),
    "race as reason":       ("race_reason",                "race_influence"),
    "appearance mentioned": ("appearance_mentioned_human", "appearance_mentioned_judge"),
    "appearance as reason": ("appearance_reason",          "appearance_influence"),
    "contradicted":         ("contradicted_human",         "contradicted_judge"),
}


def to01(s, blank_as_zero=False):
    """Convert labels (0/1, 1.0/0.0, True/False, yes/no) to 0/1.
    blank_as_zero=True: empty cells count as 0 (used for the human labels)."""
    m = {"1": 1, "0": 0, "1.0": 1, "0.0": 0, "true": 1, "false": 0,
         "yes": 1, "no": 0, "y": 1, "n": 0}
    text = s.astype(str).str.strip().str.lower()
    out = text.map(m)
    if blank_as_zero:
        blank = s.isna() | (text == "") | (text == "nan")
        out = out.where(~blank, 0)
    return out


rows = []
print("=== Confusion counts (human vs judge) ===")
for name, (hc, jc) in PAIRS.items():
    if hc not in df.columns or jc not in df.columns:
        print(f"SKIP {name}: column missing ({hc} or {jc})")
        continue

    h = to01(df[hc], blank_as_zero=True)   # your labels: blank = 0
    j = to01(df[jc])                        # judge labels: blank = not evaluated
    ok = h.notna() & j.notna()
    if (~ok).any():
        print(f"WARNING: {name}: {(~ok).sum()} rows excluded (unreadable label or judge not evaluated)")
    h, j = h[ok].astype(int), j[ok].astype(int)

    agree = (h == j).mean() * 100
    kappa = cohen_kappa_score(h, j) if (h.nunique() > 1 or j.nunique() > 1) else np.nan

    row = {
        "label": name,
        "n": len(h),
        "agreement_%": round(agree, 1),
        "kappa": round(kappa, 2) if not np.isnan(kappa) else "n/a",
        "human_pos": int(h.sum()),
        "judge_pos": int(j.sum()),
    }
    if name == "contradicted":
        row["precision"] = round(precision_score(h, j, zero_division=0), 2)
        row["recall"] = round(recall_score(h, j, zero_division=0), 2)
    rows.append(row)

    tn, fp, fn, tp = confusion_matrix(h, j, labels=[0, 1]).ravel()
    print(f"{name:22s} TP={tp:3d}  FP={fp:3d}  FN={fn:3d}  TN={tn:3d}")

res = pd.DataFrame(rows)
print("\n=== Agreement summary ===")
print(res.to_string(index=False))
res.to_csv("audit_agreement.csv", index=False)

# Rows where you and the judge disagree, to look at if a number seems off
disagree = []
for name, (hc, jc) in PAIRS.items():
    if hc in df.columns and jc in df.columns:
        h = to01(df[hc], blank_as_zero=True)
        j = to01(df[jc])
        mask = h.notna() & j.notna() & (h != j)
        for i in df.index[mask]:
            disagree.append({"id": df.at[i, "id"], "label": name,
                             "human": int(h[i]), "judge": int(j[i]),
                             "model": df.at[i, "model"], "vignette_id": df.at[i, "vignette_id"]})
pd.DataFrame(disagree).to_csv("audit_disagreements.csv", index=False)
print(f"\nSaved {len(disagree)} disagreements to audit_disagreements.csv")





# r = s[s.id == 103].iloc[0]
# print(r[["model", "prompt", "vignette_id", "photo_id", "response_n"]])
# print("JUDGE QUOTE:", r["gender_quote"])
# print("JUDGE REASONING:", r["gender_reasoning"])
# print("\nFULL EXPLANATION:\n", load_reasoning(r))



# import json
# import pandas as pd

# dis = pd.read_csv("audit_disagreements.csv")
# ids = dis.loc[dis.label == "contradicted", "id"].tolist()

# with open("contradiction_review.txt", "w") as out:
#     for i in ids:
#         r = s[s.id == i].iloc[0]
#         probs = json.loads(r.fidelity_problems) if isinstance(r.fidelity_problems, str) else []
#         contra = [p for p in probs if p.get("type") == "CONTRADICTED"]

#         out.write("=" * 100 + "\n")
#         out.write(f"ID {i} | {r.model} | {r.prompt} | V{int(r.vignette_id):02d} | {r.photo_id} | {r.response_n}\n")
#         out.write("-" * 100 + "\n")
#         out.write("JUDGE FLAGS (contradicted):\n")
#         for k, p in enumerate(contra, 1):
#             out.write(f"  [{k}] EXPLANATION: {p.get('quote')}\n")
#             out.write(f"      CASE TEXT:   {p.get('case_text_quote')}\n")
#         out.write("\nFULL EXPLANATION:\n")
#         out.write(sheet.loc[sheet.id == i, "explanation"].iloc[0] + "\n")
#         out.write("\nCASE TEXT:\n")
#         out.write(sheet.loc[sheet.id == i, "case_text"].iloc[0] + "\n\n")

# print(f"Wrote {len(ids)} cases to contradiction_review.txt")