import json
import shutil
import os

BASE = os.path.join(os.path.dirname(__file__), "../15_Apr/vignettes_nature_paper")
BASE = os.path.abspath(BASE)

MAIN_FILE   = os.path.join(BASE, "ground_truth_mapping.json")
NEW_FILE    = os.path.join(BASE, "ground_truth_mapping_new_indian.json")
BACKUP_FILE = os.path.join(BASE, "ground_truth_mapping_before_indian_swap.json")

# Which old slot maps to which new_indian slot
# (new_indian uses its own sequential numbering within each subgroup)
SLOT_MAP = {
    "BM_4": "BM_1",
    "IF_1": "IF_1",
    "IF_2": "IF_2",
    "IF_3": "IF_3",
    "IF_4": "IF_4",
    "IM_1": "IM_1",
    "IM_3": "IM_2",
    "IM_5": "IM_3",
}

with open(MAIN_FILE) as f:
    main = json.load(f)
with open(NEW_FILE) as f:
    new_ind = json.load(f)

# Back up before touching anything
shutil.copy(MAIN_FILE, BACKUP_FILE)
print(f"Backup saved to {os.path.basename(BACKUP_FILE)}")

updated = 0
for key in main:
    # key looks like "01_BM_4" — extract the face slot part
    parts = key.split("_", 1)   # ["01", "BM_4"]
    if len(parts) != 2:
        continue
    vignette, face_slot = parts

    if face_slot not in SLOT_MAP:
        continue

    new_slot = SLOT_MAP[face_slot]
    new_key = f"{vignette}_{new_slot}"   # e.g. "01_BM_1"
    new_encoded = new_ind.get(new_key)

    if new_encoded is None:
        print(f"WARNING: {new_key} not found in new_indian mapping")
        continue

    old_encoded = main[key]
    main[key] = new_encoded
    print(f"  {key}: {old_encoded} → {new_encoded}")
    updated += 1

with open(MAIN_FILE, "w") as f:
    json.dump(main, f, indent=4)

print(f"\nDone. Updated {updated} entries in {os.path.basename(MAIN_FILE)}")
print(f"Expected: 12 vignettes × {len(SLOT_MAP)} slots = {12 * len(SLOT_MAP)}")
