import os
import json
import pandas as pd

# ── Paths ──
RESULTS_DIR = "../15_Apr/vignettes_nature_paper/results_qwen25_cfd"
OUTPUT_EXCEL = "../15_Apr/vignettes_nature_paper/results_qwen25_cfd/Master_Regression_Dataframe.xlsx"
MAPPING_FILE = "../15_Apr/vignettes_nature_paper/ground_truth_mapping.json"
CFD_METADATA_FILE = "../images/cfd_metadata.xlsx" 

PROMPT_TYPES = ["prompt_baseline", "prompt_ignore", "prompt_acknowledge"]
SUBGROUP_ORDER = ["AF", "AM", "BF", "BM", "IF", "IM", "LF", "LM", "WF", "WM"]

# ── 1. Load CFD Metadata (EXPANDED) ────────────────────────────────
print("Loading CFD metadata...")
cfd_meta = pd.read_excel(CFD_METADATA_FILE, skiprows=3, header=None)

# We are now grabbing all the useful columns from the CFD file
cols_to_keep = [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
cfd_meta = cfd_meta[cols_to_keep]

# Rename them to beautiful, clean headers
cfd_meta.columns = [
    'photo_id', 'cfd_name', 'Rated_Age', 
    'FemaleProb', 'MaleProb', 'BlackProb', 'LatinoProb', 'OtherProb', 'WhiteProb',
    'Afraid', 'Angry', 'Disgusted', 'Happy', 'Sad', 'Surprised',
    'Attractive', 'Babyfaced', 'Feminine', 'Masculine', 'Prototypic', 
    'Threatening', 'Trustworthy', 'Unusual', 'Luminance_median'
]

with open(MAPPING_FILE) as f:
    mapping = json.load(f)
inv_map = {v: k for k, v in mapping.items()}

# ── 2. Load Data at the RUN LEVEL (0 or 1) ─────────────────────────
master_rows = []

for prompt_type in PROMPT_TYPES:
    prompt_dir = os.path.join(RESULTS_DIR, prompt_type)
    if not os.path.isdir(prompt_dir): continue

    print(f"Parsing {prompt_type}...")
    for encoded_name in os.listdir(prompt_dir):
        human_name = inv_map.get(encoded_name)
        if not human_name: continue

        parts = human_name.split("_")
        if len(parts) != 3: continue 

        vignette_id, subgroup, run_num = parts
        if subgroup not in SUBGROUP_ORDER: continue

        photo_id = f"{subgroup}_{run_num}"
        # vignette_class = 'Borderline' if vignette_id in ['03', '08', '09', '11', '12', '02', '06', '07', '01'] else 'Clear'
        vignette_class = 'Borderline' if vignette_id in ['03', '08', '09', '11', '12'] else 'Clear'

        run_dir = os.path.join(prompt_dir, encoded_name)
        for file in os.listdir(run_dir):
            if not file.endswith(".json"): continue
            
            try:
                with open(os.path.join(run_dir, file)) as f:
                    data = json.load(f)
            except json.JSONDecodeError: continue
                
            ans = data.get("parsed_answer", "")
            if ans in ["admit", "discharge"]:
                master_rows.append({
                    "model": "qwen25",
                    "prompt": prompt_type,
                    "vignette_id": vignette_id,
                    "vignette_class": vignette_class,
                    "photo_id": photo_id,
                    "race": subgroup[0],
                    "gender": subgroup[1],
                    "admit_decision": 1 if ans == "admit" else 0
                })

df_master = pd.DataFrame(master_rows)

# ── 3. Merge Metadata and Z-Score EVERYTHING ───────────────────────
df_master = pd.merge(df_master, cfd_meta, on='photo_id', how='left')

# Grab all the CFD continuous columns we just attached (skipping the first 2 IDs)
continuous_features = cfd_meta.columns[2:]

# Z-score all of them for the regression
for feature in continuous_features:
    df_master[f'{feature}_z'] = (df_master[feature] - df_master[feature].mean()) / df_master[feature].std()

# ── 4. Save to Disk ────────────────────────────────────────────────
df_master.to_excel(OUTPUT_EXCEL, index=False)
df_master.to_csv(OUTPUT_EXCEL.replace('.xlsx', '.csv'), index=False)

print(f"\nDone! Master dataframe created with {len(df_master)} rows.")
print(f"Included {len(continuous_features)} continuous CFD features.")
print(f"Saved to {OUTPUT_EXCEL}")