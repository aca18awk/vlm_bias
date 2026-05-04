import os
import json
import pandas as pd

# ── Paths & Setup ──
MODEL_DIRS = {
    "llama": "../15_Apr/vignettes_nature_paper/results_llama32_cfd",
    "qwen": "../15_Apr/vignettes_nature_paper/results_qwen25_cfd" 
}
MAPPING_FILE = "../15_Apr/vignettes_nature_paper/ground_truth_mapping.json"
CFD_METADATA_FILE = "../images/cfd_metadata.xlsx" 

PROMPT_TYPES = ["prompt_baseline", "prompt_ignore", "prompt_acknowledge"]
SUBGROUP_ORDER = ["AF", "AM", "BF", "BM", "IF", "IM", "LF", "LM", "WF", "WM"]

# ── 1. Load & Clean CFD Metadata ───────────────────────────────────
print("Loading CFD metadata...")
cfd_meta = pd.read_excel(CFD_METADATA_FILE, skiprows=3, header=None)

cols_to_keep = [0, 1, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
cfd_meta = cfd_meta[cols_to_keep]

cfd_meta.columns = [
    'photo_id', 'cfd_name', 'age_rated', 
    'female_prob', 'BlackProb', 'LatinoProb', 'OtherProb', 'WhiteProb',
    'afraid', 'angry', 'disgusted', 'happy', 'sad', 'surprised',
    'attractive', 'babyfaced', 'feminine', 'masculine', 'prototypic', 
    'threatening', 'trustworthy', 'Unusual', 'luminance'
]

# Calculate race_ambiguity
cfd_meta['race_ambiguity'] = 1 - cfd_meta[['BlackProb', 'LatinoProb', 'OtherProb', 'WhiteProb']].max(axis=1)

print(f"CFD metadata loaded: {len(cfd_meta)} faces, {len(cfd_meta.columns)} columns")
print(f"photo_ids: {sorted(cfd_meta['photo_id'].tolist())}")

with open(MAPPING_FILE) as f:
    mapping = json.load(f)
inv_map = {v: k for k, v in mapping.items()}

# ── 2. Load Data at the RUN LEVEL ──────────────────────────────────
master_rows = []

for model_name, results_dir in MODEL_DIRS.items():
    if not os.path.isdir(results_dir):
        print(f"Warning: Directory not found for {model_name} at {results_dir}")
        continue

    rows_before = len(master_rows)
    print(f"Parsing inference data for {model_name}...")
    for prompt_dir_name in PROMPT_TYPES:
        prompt_path = os.path.join(results_dir, prompt_dir_name)
        if not os.path.isdir(prompt_path): continue

        clean_prompt = prompt_dir_name.replace("prompt_", "")

        for encoded_name in os.listdir(prompt_path):
            human_name = inv_map.get(encoded_name)
            if not human_name: continue

            parts = human_name.split("_")
            if len(parts) != 3: continue 

            vignette_str, subgroup, run_num = parts
            if subgroup not in SUBGROUP_ORDER: continue

            # --- FIXED INDENTATION AND MISSING VARIABLE BELOW ---
            photo_id = f"{subgroup}_{run_num}"
            vignette_id = int(vignette_str)  # Restored this line!
            borderline_ids = [3, 8, 9, 11, 12] if model_name == "qwen" else [3, 8, 9, 11, 12, 2, 6, 7, 1]
            vignette_class = 'borderline' if vignette_id in borderline_ids else 'clear'

            run_dir = os.path.join(prompt_path, encoded_name)
            
            json_files = sorted([f for f in os.listdir(run_dir) if f.endswith(".json")])
            
            for file in json_files:
                try:
                    with open(os.path.join(run_dir, file)) as f:
                        data = json.load(f)
                except json.JSONDecodeError: continue
                    
                ans = data.get("parsed_answer", "")
                if ans in ["admit", "discharge", "**admit**", "**discharge**",  "*admit*", "*discharge*"]:
                    master_rows.append({
                        "model": model_name,
                        "prompt": clean_prompt,
                        "vignette_id": vignette_id,
                        "vignette_class": vignette_class,
                        "photo_id": photo_id,
                        "race": subgroup[0],
                        "gender": subgroup[1],
                        "response_n": file, 
                        "admit_decision": 1 if ans in ["admit",  "**admit**",  "*admit*"] else 0
                    })
            # --- END OF FIXES ---
    print(f"  → {len(master_rows) - rows_before} run-level rows collected for {model_name}")

df_run_level = pd.DataFrame(master_rows)
print(f"\nRun-level dataframe: {df_run_level.shape[0]} rows × {df_run_level.shape[1]} cols")

# ── 3. Z-Score CFD features across the 50 unique faces, then merge ──
# Z-scores computed on cfd_meta (50 rows) so mean/std reflect the face
# distribution, not the repeated run-level data.
print("Z-scoring predictors and merging metadata...")
features_to_zscore = [col for col in cfd_meta.columns if col not in ['photo_id', 'cfd_name']]

for feature in features_to_zscore:
    cfd_meta[f'{feature}_z'] = (cfd_meta[feature] - cfd_meta[feature].mean()) / cfd_meta[feature].std()

df_run_level = pd.merge(df_run_level, cfd_meta, on='photo_id', how='left')
unmatched = df_run_level['age_rated'].isna().sum()
if unmatched:
    print(f"WARNING: {unmatched} rows have no CFD metadata match — unrecognised photo_ids:")
    print(df_run_level[df_run_level['age_rated'].isna()]['photo_id'].unique())
else:
    print("Metadata merge OK — all photo_ids matched.")

# ── 4. Create the Aggregated Version ───────────────────────────────
print("Building aggregated dataset...")

# Dynamically build the grouping columns so no metadata gets dropped
group_cols = ['model', 'vignette_id', 'vignette_class', 'prompt', 'photo_id', 'cfd_name', 'race', 'gender'] + \
             features_to_zscore + [f'{f}_z' for f in features_to_zscore]

df_aggregated = df_run_level.groupby(group_cols).agg(
    n_admits=('admit_decision', 'sum'),
    n_total=('admit_decision', 'count')
).reset_index()

df_aggregated['admit_rate'] = df_aggregated['n_admits'] / df_aggregated['n_total']

# ── 5. Save Both to Disk ───────────────────────────────────────────
df_run_level.to_csv("Master_RunLevel_Data.csv", index=False)
df_aggregated.to_csv("Master_Aggregated_Data.csv", index=False)

print(f"\nDone! Pipeline complete.")
print(f"Created Run-Level CSV/Parquet: {len(df_run_level)} rows.")
print(f"Created Aggregated CSV/Parquet: {len(df_aggregated)} rows.")

print("\n── Aggregated dataframe diagnostics ──")
print(f"Shape: {df_aggregated.shape}")

incomplete = df_aggregated[df_aggregated['n_total'] < 20]
print(f"\nCells with < 20 runs: {len(incomplete)}" + (f" ← WARNING" if len(incomplete) else " ✓"))
if len(incomplete):
    print(incomplete[['model', 'prompt', 'photo_id', 'vignette_id', 'n_total']].to_string())

print(f"\nn_total distribution: min={df_aggregated['n_total'].min()}  max={df_aggregated['n_total'].max()}  mean={df_aggregated['n_total'].mean():.1f}")

print("\nHead:")
print(df_aggregated.head().to_string())
print("\nCell sizes (model × prompt × race × gender):")
print(df_aggregated.groupby(['model', 'prompt', 'race', 'gender']).size().to_string())