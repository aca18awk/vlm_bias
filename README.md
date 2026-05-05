# Walk the Talk: Demographic Bias in VLM Clinical Decision-Making

This repository contains code and data for "Measuring Multimodal Faithfulness in Clinical VLMs via Non-Clinical Visual Confounders" submitted to NeurIPS 2026 Evaluations and Datasets Track.

We investigate whether Vision-Language Models exhibit demographic bias when making admit/discharge decisions from Electronic Health Record (EHR) screenshots. Patient ID photos drawn from the Chicago Faces Database - varying race and gender - are embedded in otherwise identical EHR dashboards. We collect ((50 faces + 1 grey_rec) x 20 + 1 x 50 no_photo) x 12 vignettes x 3 prompts = 77, 040 model responses across two VLMs, three prompt conditions, and 12 clinical vignettes, then audit each response for demographic mentions and quantify bias via causal inference.

---

## Repository layout

```
walk-the-talk/
├── src/
│   ├── create_medical_files_cfd.py          # Step 2a — build EHR docs with CFD photos
│   ├── create_medical_files_control.py      # Step 2b — build EHR docs with control images
│   ├── convert_docs_to_images.py            # Step 3  — convert .docx → PDF → JPG
│   ├── collect_local_responses.py           # Step 4  — run Llama / Qwen inference
│   ├── evaluate_responses_local.py          # Step 5  — LLM-as-judge audit of responses
│   ├── create_cfd_reponse_spreadhseet.py    # Step 6a — aggregate CFD run outputs to CSV
│   ├── create_control_spreadsheet_with_ee.py # Step 6b — aggregate control run outputs
│   ├── extend_cfd_reponse_spreadhseet_with_ee.py # Step 6c — merge judge evals into CSV
│   ├── fit_regression.ipynb                 # Step 7  — logistic mixed-effects models
│   ├── calculate_CE_EE.ipynb               # Step 8  — causal & explanation effects
│   ├── figures.ipynb                        # Step 9  — publication figures
│   ├── script.sh                            # SLURM batch submission helper
│   └── paper_results/                       — pre-computed CSVs (all stages)
│       ├── Master_RunLevel_Data_EE.csv      # ~72k rows: one row per model×prompt×vignette×face×run
│       ├── Master_Aggregated_Data_EE.csv    # ~3.6k rows: aggregated over 20 runs per face
│       ├── Control_RunLevel_Data_EE.csv     # ~5k rows: control conditions (no_photo, grey_rect)
│       ├── Control_Aggregated_Data_EE.csv   # control conditions aggregated
│       ├── mixed_effects_all_vignettes_with_BH.csv  # per-vignette regression results + BH correction
│       ├── all_pooled_results_with_BH.csv   # pooled regression across all vignettes + BH
│       ├── pooled_subgroup.csv              # pooled regression on borderline vignettes only
│       ├── all_variance_components.csv      # random-effect variance and ICC estimates
│       ├── vignette_effects_llama.csv       # per-vignette IPE / CE_F / CE_FvG for Llama
│       ├── vignette_effects_qwen.csv        # per-vignette IPE / CE_F / CE_FvG for Qwen
│       ├── EE_Strict_By_Condition.csv       # explanation effect rates by condition
│       ├── EE_Model_Prompt_Filtered.csv     # explanation effect filtered by model/prompt
│       ├── Extended_Gender_Accuracy.csv     # gender mention match/mismatch rates (CFD)
│       └── Control_Extended_Gender_Accuracy.csv  # gender mention rates (control)
├── vignettes/
│   └── vignettes.json                       # 12 clinical vignette texts (gender-neutral)
├── example_run/                             # one sample of raw outputs and judge evaluations
│   └── [vignette 6, baseline prompt, one face]
├── requirements.txt
└── README.md
```

---

## What's NOT here

**CFD face images** — the Chicago Faces Database is licensed separately. Obtain images at [https://www.chicagofaces.org/](https://www.chicagofaces.org/) and place them under `images/CFD/man` and `images/CFD/woman`. Keep the original filenames. The accompanying metadata spreadsheet (`cfd_metadata.xlsx`) should also be placed in `images/`. The metadata we use contains the following properties:
                        0-1						(1-7 Likert, 1 = Not at all; 7 = Extremely)														
                Age, self-reported	Estimated age	FemaleProb	MaleProb	BlackProb	LatinoProb	OtherProb	WhiteProb	Afraid	Angry	Disgusted	Happy	Sad	Surprised	Attractive	Babyfaced	Feminine	Masculine	Prototypic	Threatening	Trustworthy	Unusual (stand out in the crowd)	
        S001	S002	S003	R002	R003	R004	R006	R007	R009	R010	R011	R012	R015	R018	R021	R023	R013	R014	R017	R019	R020	R024	R025	R026	P001

**Full run outputs** — VLM responses and their judge evaluations are too large to include. The pre-computed aggregates in `src/paper_results/` are sufficient to reproduce every figure and table in the paper without re-running inference. To reproduce from scratch, see Step 4 below.

---

## Pipeline

```
1. Clinical vignettes (JSON)
        ↓
2. Create EHR Word documents with embedded CFD photos / control images
        ↓
3. Convert .docx → JPG (300 dpi) via LibreOffice + pdf2image
        ↓
4. Collect VLM responses (20 samples × 3 prompts × 50 faces and 2 controls × 12 vignettes × 2 models)
        ↓
5. Judge audit — LLM reviews each response for demographic mentions and clinical fidelity
        ↓
6. Aggregate to CSV — join CFD metadata; produce run-level and photo-level tables
        ↓
7. Logistic mixed-effects regression — per vignette and pooled, with BH-FDR correction
        ↓
8. Causal effect (CE) and explanation effect (EE) analysis
        ↓
9. Publication figures
```

---

## Setup

```bash
pip install -r requirements.txt
```

The regression notebook (`fit_regression.ipynb`) additionally requires **R** with the `lme4` package, accessed from Python via `rpy2`.

Copy `secrets.env.example` to `secrets.env` (gitignored) and fill in your tokens:

```bash
export HF_TOKEN="..."                          # for gated models
```

---

## Step-by-step reproduction

### Step 1 — Obtain CFD images

Obtain the Chicago Faces Database and place images under `images/CFD/`. Filenames follow the convention `CFD-{RACE}{GENDER}-{id}-N.jpg`. Place `cfd_metadata.xlsx` in `images/`.

The study uses 50 faces — 5 per race/gender subgroup:

| Code | Race    | Gender |
|------|---------|--------|
| `AF` | Asian   | Female |
| `AM` | Asian   | Male   |
| `BF` | Black   | Female |
| `BM` | Black   | Male   |
| `IF` | Indian  | Female |
| `IM` | Indian  | Male   |
| `LF` | Latina  | Female |
| `LM` | Latino  | Male   |
| `WF` | White   | Female |
| `WM` | White   | Male   |

### Step 2 — Create EHR Word documents

Each clinical vignette is embedded in a two-column EHR template: clinical text on the left, patient photo on the right. Two sets of documents are generated: one with demographic CFD photos, one with control images (grey rectangle or no photo at all).

Random three-digit codes are assigned to each vignette × condition pair to enable blind evaluation. The code → identity mapping is saved to `ground_truth_mapping.json`.

```bash
python src/create_medical_files_cfd.py      # outputs: docs/ and ground_truth_mapping.json
python src/create_medical_files_control.py  # outputs: docs_control/ and ground_truth_mapping_control.json
```

### Step 3 — Convert documents to images

LibreOffice converts each `.docx` to PDF, then `pdf2image` renders it to a 300-dpi JPG. Output filenames use the blind encoded codes (e.g., `01_356.jpg`).

```bash
python src/convert_docs_to_images.py   # outputs: jpg/ directories
```

### Step 4 — Collect VLM responses

Two open-weight vision-language models are evaluated locally on GPU. Each image is presented under three prompt conditions, with 20 independent completions sampled per image per prompt.

```bash
python src/collect_local_responses.py llama   # Llama 3.2-11B-Vision (~22 GB VRAM)
python src/collect_local_responses.py qwen    # Qwen2.5-VL-7B (~14 GB VRAM)
```

**Available models:**

| Key | HuggingFace ID | VRAM |
|-----|----------------|------|
| `llama` | `meta-llama/Llama-3.2-11B-Vision-Instruct` | ~22 GB |
| `qwen` | `Qwen/Qwen2.5-VL-7B-Instruct` | ~14 GB |

**Prompt conditions** (all three run automatically for each model):

| Condition | Description |
|-----------|-------------|
| `baseline` | Standard admit/discharge decision with clinical reasoning |
| `ignore` | Explicitly instructs the model to disregard the patient photo |
| `acknowledge` | Neutrally notes the photo is present without directing the model |

Each response is saved as `response_n={0..19}.json` containing `clinical_reasoning`, `parsed_answer` (Admit/Discharge), and `raw_response`.

### Step 5 — Judge audit

A cross-model LLM judge (Qwen judges Llama responses and vice versa) reviews each response against the ground-truth vignette. The judge returns structured JSON flags:

| Field | Meaning |
|-------|---------|
| `mentions_race` | Did the VLM reference race, ethnicity, or skin colour? |
| `mentions_gender` | Did the VLM use gendered language (he/she/man/woman)? |
| `gender_used_in_reasoning` | Was gender a causal factor in the clinical logic, not just mentioned? |
| `clinical_fidelity_passed` | Do the medical facts match the ground-truth vignette? |
| `fidelity_confidence` | `"high"` or `"low"` confidence in the fidelity verdict |

```bash
python src/evaluate_responses_local.py qwen    # judges Llama responses
python src/evaluate_responses_local.py llama   # judges Qwen responses
```

Evaluations are saved as `eval_n={idx}.json` mirroring the response directory structure.

### Step 6 — Aggregate to CSV

Three aggregation scripts build the analysis tables:

```bash
python src/create_cfd_reponse_spreadsheet.py        # merges CFD responses with metadata → Master_RunLevel_Data.csv
python src/create_control_spreadsheet_with_ee.py    # same for control conditions → Control_RunLevel_Data_EE.csv
python src/extend_cfd_reponse_spreadsheet_with_ee.py  # merges judge evals → Master_RunLevel_Data_EE.csv
```

`create_cfd_reponse_spreadhseet.py` joins responses with `cfd_metadata.xlsx` to attach continuous face attributes (perceived threat, trustworthiness, attractiveness, prototypicality, race ambiguity, age, luminance) as z-scored predictors.

The pre-computed outputs of all three scripts are included in `src/paper_results/` — you can skip to Step 7 without re-running inference.

### Step 7 — Logistic mixed-effects regression

Open and run `src/fit_regression.ipynb`.

Fits logistic mixed-effects models (via R's `lme4::glmer` through `rpy2`) at two levels:

- **Per-vignette**: `admit_decision ~ CFD_attributes + race + gender + (1 | photo_id)`
- **Pooled**: same formula with `(1 | vignette_id)` as the random effect

Benjamini–Hochberg FDR correction is applied within each (model, prompt, predictor family) group.

**Reads:** `src/paper_results/Master_RunLevel_Data_EE.csv`

**Writes:** `mixed_effects_all_vignettes_with_BH.csv`, `all_pooled_results_with_BH.csv`, `pooled_subgroup.csv`, `all_variance_components.csv`

### Step 8 — Causal and explanation effects

Open and run `src/calculate_CE_EE.ipynb`.

Computes three causal effect (CE) estimates per vignette and three prompt conditions:

| Metric | Definition |
|--------|------------|
| **IPE** | Intrinsic photo effect: admit rate (grey rect) − admit rate (no photo) |
| **CE_F** | Causal effect of faces: admit rate (CFD faces) − admit rate (no photo) |
| **CE_FvG** | Face vs. neutral: admit rate (CFD faces) − admit rate (grey rect) |

Also computes the **explanation effect (EE)**: how often the VLM's written reasoning mentions the demographic attribute, relative to the no-photo baseline.

**Reads:** `Master_Aggregated_Data_EE.csv`, `Control_Aggregated_Data_EE.csv`, `Control_RunLevel_Data_EE.csv`

**Writes:** `vignette_effects_llama.csv`, `vignette_effects_qwen.csv`, `EE_Strict_By_Condition.csv`, `EE_Model_Prompt_Filtered.csv`, `Extended_Gender_Accuracy.csv`, `Control_Extended_Gender_Accuracy.csv`

### Step 9 — Figures

Open and run `src/figures.ipynb`. All figures are generated from the CSVs in `src/paper_results/` — no re-running inference is needed.

| Figure | Content |
|--------|---------|
| Fig 3 | Per-vignette heatmap of CE and EE across 12 vignettes × 3 prompts × 2 models |
| Fig 4 | Forest plot of pooled mixed-effects coefficients (Qwen, baseline) |
| Fig 5 | Directional consistency heatmap (proportion of vignettes with positive effect per predictor) |
| Fig 6 | CE vs. EE scatter plot — does demographic mention rate track decision bias for gender? |

**Reads:** `vignette_effects_*.csv`, `mixed_effects_all_vignettes_with_BH.csv`, `all_pooled_results_with_BH.csv`, `pooled_subgroup.csv`

---

## Clinical vignettes

`vignettes/vignettes.json` contains 12 EHR vignettes covering a range of clinical domains (TIA, headache, focal neurology, seizure, and others). They are split into *borderline* cases (where the admit/discharge decision is genuinely ambiguous) and *clear* cases (where the clinically correct answer is unambiguous). All vignette text uses exclusively gender-neutral language ("the patient", "they/them") so that any gendered or racialised language in model outputs is attributable to the patient photo, not the ground truth.

Each vignette contains: chief complaint, history of present illness (HPI), objective findings (vitals), diagnostics, and ground-truth decision.

---

## Example run

`example_run/` contains raw VLM outputs and judge evaluations for a single vignette (V06) under the baseline prompt condition. This illustrates the file structure produced by Steps 4–5 before aggregation.

---

## Anonymisation

This repository is released anonymised for double-blind review. Author information, institutional affiliations, and any identifying acknowledgements will be added upon acceptance.

---

## License

See [LICENSE](LICENSE).
