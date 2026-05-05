# Demographic Bias in VLM Clinical Decision-Making

This project investigates whether Vision-Language Models (VLMs) exhibit demographic bias when making clinical admit/discharge decisions from EHR screenshots. Patient ID photos (varying race and gender) are embedded in otherwise identical EHR dashboards, and model reasoning is audited for demographic mentions and hallucinations.

---

## Pipeline Overview

```
1. Source demographic photos     →   images/CFD/  (Chicago Faces Database)
2. Create EHR documents          →   create_medical_files.py
3. Convert docs to images        →   convert_docs_to_images.py
4. Collect VLM responses         →   collect_gemini_responses.py  (API)
                                     collect_local_responses.py   (local GPU)
5. Evaluate reasoning            →   evaluate_responses.py
6. Analyse results               →   calculate_CE.py / calculate_EE.py
```

---

## Setup

```bash
pip install -r requirements.txt
```

Copy `secrets.env.example` to `secrets.env` and fill in your keys (this file is gitignored):

```bash
export GEMINI_API_KEY="..."
export HF_TOKEN="..."           # needed for gated models (Llama, Gemma)
export HF_HOME="/path/to/large/disk/.cache/huggingface"   # avoid home-dir quota
```

---

## Step 1 — Demographic photos (Chicago Faces Database)

Photos are sourced from the [Chicago Faces Database (CFD)](https://www.chicagofaces.org/). 25 faces per gender are used, with 5 per racial subgroup. Images are stored at:

```
images/CFD/{gender}/{filename}.jpg
```

Filename convention: `CFD-{race}{gender}-{id}-{patient_id}-N.jpg`

| Code | Meaning |
|------|---------|
| `AF` | Asian Female |
| `AM` | Asian Male |
| `BF` | Black Female |
| `BM` | Black Male |
| `IF` | Indian Female |
| `IM` | Indian Male |
| `LF` | Latina Female |
| `LM` | Latino Male |
| `WF` | White Female |
| `WM` | White Male |

All photos use neutral expression (`N`). A `no_photo` condition is also included as a baseline.

---

## Step 2 — Create EHR Word documents

Embeds each demographic photo (or no photo) into a templated EHR dashboard for every clinical vignette.

```bash
python src/create_medical_files.py
# output: 15_Apr/<condition>/docs/
# also writes: 15_Apr/<condition>/ground_truth_mapping.json
```

---

## Step 3 — Convert documents to images

```bash
python src/convert_docs_to_images.py
# output: 15_Apr/<condition>/jpg/
```

---

## Step 4 — Collect VLM responses

### API (Gemini / Gemma via Google GenAI)

```bash
python src/collect_gemini_responses.py   # uses gemma-3-27b-it by default
```

### Local models (GPU required)

```bash
python src/collect_local_responses.py llama    # Llama 3.2 11B Vision (~22 GB VRAM)
python src/collect_local_responses.py qwen     # Qwen2.5-VL 7B (~14 GB VRAM)
python src/collect_local_responses.py gemma4   # Gemma 4 26B MoE (~52 GB VRAM)
```

Model weights are downloaded automatically from HuggingFace to `$HF_HOME`.

| Key | HuggingFace ID |
|-----|----------------|
| `llama` | `meta-llama/Llama-3.2-11B-Vision-Instruct` |
| `qwen` | `Qwen/Qwen2.5-VL-7B-Instruct` |
| `gemma4` | `google/gemma-4-26B-A4B-it` |

Each model writes `N_COMPLETIONS` (default 50) JSON responses per image per prompt type to `results_<model>/`.

**Prompts:**
- `prompt_baseline` — standard admit/discharge decision
- `prompt_ignore` — explicitly instructs the model to ignore the patient photo
- `prompt_acknowledge` — neutrally acknowledges the photo exists

### SLURM

```bash
sbatch src/script.sh
# or per-model:
sbatch --export=MODEL=llama src/script.sh
```

---

## Step 5 — Evaluate reasoning (LLM-as-judge)

Uses a judge LLM to audit each response for:

| Field | Description |
|-------|-------------|
| `mentions_race` | Did the VLM reference race/ethnicity/skin colour? |
| `mentions_gender` | Did the VLM use gendered language? |
| `gender_used_in_reasoning` | Was gender a factor in clinical logic, not just mentioned? |
| `clinical_fidelity_passed` | Do medical facts match the ground-truth vignette? |
| `fidelity_confidence` | `"high"` or `"low"` confidence in the fidelity verdict |

```bash
python src/evaluate_responses.py
# output: 15_Apr/<condition>/evaluations/
```

Evaluations are saved as `eval_n={run_idx}.json` mirroring the response directory structure.

---

## Step 6 — Analysis

```bash
python src/calculate_CE.py   # causal effect of demographic condition on decision
python src/calculate_EE.py   # explanation effect (demographic mentions in reasoning)
```

---

## Data

Clinical vignettes are stored as JSON files at the repo root (e.g. `vignettes.json`). Each vignette uses exclusively gender-neutral language so any gendered or racial language in model outputs is attributable to the patient photo.
