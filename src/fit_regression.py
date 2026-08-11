
import pandas as pd
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
lme4 = importr('lme4')


AGGREDATED_DATA = "Master_Aggregated_Data_Medgemma_EE.csv"
RUN_DATA = "Master_RunLevel_Data_Medgemma_EE.csv"

print("Loading run-level CSV...")
df = pd.read_csv(RUN_DATA)



def fit_vignette(df, model_name, prompt_name, vid):
    # 1. Slice the data
    subset = df[(df.model==model_name) & (df.prompt==prompt_name) & (df.vignette_id==vid)].copy()
    
    # Safety check: if the subset is empty, skip it
    if subset.empty:
        return None

    formula_r = ro.Formula(
        "admit_decision ~ threatening_z + trustworthy_z + attractive_z + "
        "prototypic_z + race_ambiguity_z + age_rated_z + "
        "luminance_z + race + gender + (1 | photo_id)"
    )
    
    # 2. Try to run the model (Safety Net for non-converging data!)
    try:
        with localconverter(ro.default_converter + numpy2ri.converter + pandas2ri.converter):
            r_df = ro.conversion.py2rpy(subset)
            
        fit = lme4.glmer(formula_r, data=r_df, family=ro.r('binomial()'))
        coef_table = ro.r('function(m) as.data.frame(summary(m)$coefficients)')(fit)
        
        with localconverter(ro.default_converter + numpy2ri.converter + pandas2ri.converter):
            coef_df = ro.conversion.rpy2py(coef_table)
            
        # 3. Rename columns to match your exact requested format
        coef_df = coef_df.rename(columns={
            'Std. Error': 'SE',
            'z value': 'z',
            'Pr(>|z|)': 'p_value'
        })
        
        # 4. Add the metadata
        coef_df['model'] = model_name
        coef_df['prompt'] = prompt_name
        coef_df['vignette'] = vid
        coef_df['predictor'] = coef_df.index
        
        # 5. Return exactly the columns you asked for, in the right order
        return coef_df[['model', 'prompt', 'vignette', 'predictor', 'Estimate', 'SE', 'z', 'p_value']]
        
    except Exception as e:
        # If R crashes on a specific vignette, print the error but keep the loop alive!
        print(f"  -> ERROR fitting {model_name} | {prompt_name} | V{vid:02d}: {e}")
        return None

# Loop over all vignettes, models, and prompts
ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
models = ["med_gemma"]
prompts = ["acknowledge", "ignore", "baseline"]

results = []
for vid in ids:
    for model in models:
        for prompt in prompts:
            print(f"Fitting {model.upper()} | Prompt: {prompt} | Vignette {vid:02d}...")
            res = fit_vignette(df, model, prompt, vid)
            
            if res is not None:
                results.append(res)

# Combine everything into one master dataframe
master_results = pd.concat(results, ignore_index=True)

# Save to a properly named CSV
master_results.to_csv("mixed_effects_all_vignettes_med_gemma.csv", index=False)

print("\n" + "="*80)
print(master_results.round(3))
print("="*80)



import pandas as pd
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr 

lme4 = importr('lme4')
utils = importr('utils')
base = importr('base')

def fit_pooled(df, model_name, prompt_name, exclude_vignettes=None):
    subset = df[(df.model==model_name) & (df.prompt==prompt_name)].copy()
    
    if exclude_vignettes:
        subset = subset[~subset.vignette_id.isin(exclude_vignettes)]
        
    if subset.empty:
        return None, None

    # No (1 | photo_id) since it had no effect
    # (1 | vignette_id) to account for baseline differences between cases
    formula_r = ro.Formula(
        "admit_decision ~ threatening_z + trustworthy_z + attractive_z + "
        "prototypic_z + race_ambiguity_z + age_rated_z + "
        "luminance_z + race + gender + (1 | vignette_id)"
    )
    
    try:
        with localconverter(ro.default_converter + numpy2ri.converter + pandas2ri.converter):
            r_df = ro.conversion.py2rpy(subset)
            
        fit = lme4.glmer(formula_r, data=r_df, family=ro.r('binomial()'))
        
        # Coefficients
        coef_table = ro.r('function(m) as.data.frame(summary(m)$coefficients)')(fit)
        with localconverter(ro.default_converter + numpy2ri.converter + pandas2ri.converter):
            coef_df = ro.conversion.rpy2py(coef_table)
            
        coef_df = coef_df.rename(columns={'Std. Error': 'SE', 'z value': 'z', 'Pr(>|z|)': 'p_value'})
        coef_df['model'] = model_name
        coef_df['prompt'] = prompt_name
        coef_df['predictor'] = coef_df.index
        
        # Variance components
        var_table = ro.r('function(m) as.data.frame(VarCorr(m))')(fit)
        with localconverter(ro.default_converter + numpy2ri.converter + pandas2ri.converter):
            var_df = ro.conversion.rpy2py(var_table)
            
        var_df['model'] = model_name
        var_df['prompt'] = prompt_name
        
        return coef_df, var_df
        
    except Exception as e:
        print(f"  -> ERROR fitting pooled {model_name} {prompt_name}: {e}")
        return None, None
        
    finally:
        # 2. FIX: Critical Garbage Collection so R doesn't crash on these massive datasets
        ro.r('gc()')



results_pooled = []
variance_components = []

qwen_without_extremes = [1,2,3,6,7,8,11,12]

for m in ['med_gemma']:
    excl = [v for v in range(1, 13) if v not in qwen_without_extremes] if m=='med_gemma' else None
    for p in ['baseline', 'ignore', 'acknowledge']:
        print(f"Fitting pooled model: {m.upper()} | Prompt: {p} ...")
        coef, var = fit_pooled(df, m, p, exclude_vignettes=excl)
        
        if coef is not None: 
            results_pooled.append(coef)
        if var is not None: 
            variance_components.append(var)

# Combine and save
pooled_df = pd.concat(results_pooled, ignore_index=True)
var_df_all = pd.concat(variance_components, ignore_index=True)

pooled_df.to_csv("all_pooled_results_med_gemma.csv", index=False)
var_df_all.to_csv("all_variance_components_med_gemma.csv", index=False)

print("\n" + "="*80)
print("SUCCESS! Pooled models completed and saved.")
print("="*80)


qwen_borderline_in_range = [3, 8, 9, 11, 12]
med_gemma_in_range = [1,2,3,6,8,11]

results_pooled_borderline = []
variance_borderline = []

for m in ['med_gemma']:
    bl = med_gemma_in_range if m=='med_gemma' else qwen_borderline_in_range
    excl = [v for v in range(1, 13) if v not in bl]
    for p in ['baseline', 'ignore', 'acknowledge']:
        print(f"Borderline-only: {m} {p}...")
        coef, var = fit_pooled(df, m, p, exclude_vignettes=excl)
        if coef is not None: 
            coef['analysis'] = 'borderline_only'
            results_pooled_borderline.append(coef)
        if var is not None:
            var['analysis'] = 'borderline_only'
            variance_borderline.append(var)

bord_df = pd.concat(results_pooled_borderline, ignore_index=True)
bord_df.to_csv("pooled_subgroup_med_gemma.csv", index=False)


import pandas as pd
import numpy as np
from scipy.stats import false_discovery_control

# Load your saved files
all_pooled = pd.read_csv('all_pooled_results_med_gemma.csv')
bord_pooled = pd.read_csv('pooled_subgroup_med_gemma.csv')


# Define predictor families
attribs = ['threatening_z','trustworthy_z','attractive_z','prototypic_z',
           'race_ambiguity_z','age_rated_z','luminance_z']
race_preds = ['raceB','raceI','raceL','raceW']
gender_preds = ['genderM']

def add_bh_correction(df):
    """Apply Benjamini-Hochberg FDR within (model, prompt, family).
    Each (model, prompt) pair has 3 families: attribute, race, gender."""
    df = df[df.predictor != '(Intercept)'].copy()
    
    # Assign family
    df['family'] = 'other'
    df.loc[df.predictor.isin(attribs), 'family'] = 'attribute'
    df.loc[df.predictor.isin(race_preds), 'family'] = 'race'
    df.loc[df.predictor.isin(gender_preds), 'family'] = 'gender'
    
    # Apply BH within each (model, prompt, family) group
    df['p_adj_m_p_f'] = np.nan
    for (m, pr, f), grp in df.groupby(['model','prompt','family']):
        df.loc[grp.index, 'p_adj_m_p_f'] = false_discovery_control(grp['p_value'].values)

        # Apply BH within each (model, prompt, family) group
    df['p_adj_m_p'] = np.nan
    for (m, pr), grp in df.groupby(['model','prompt']):
        df.loc[grp.index, 'p_adj_m_p'] = false_discovery_control(grp['p_value'].values)
    
    return df

all_pooled_bh = add_bh_correction(all_pooled)
bord_pooled_bh = add_bh_correction(bord_pooled)

# Save with BH column
all_pooled_bh.to_csv('all_pooled_results_with_BH_med_gemma.csv', index=False)
bord_pooled_bh.to_csv('pooled_subgroup_with_BH_med_gemma.csv', index=False)

# Verify what survives using the standard correction (p_adj_m_p)
print("=== Survives BH (all vignettes) ===")
print(all_pooled_bh[all_pooled_bh.p_adj_m_p < 0.05][
    ['model','prompt','predictor','Estimate','p_value', 'p_adj_m_p']
])

print("\n=== Survives BH (borderline only) ===")
print(bord_pooled_bh[bord_pooled_bh.p_adj_m_p < 0.05][
    ['model','prompt','predictor','Estimate','p_value','p_adj_m_p_f', 'p_adj_m_p']
])

print(bord_pooled_bh[bord_pooled_bh.p_value < 0.10][
    ['model','prompt','predictor','Estimate','p_value','p_adj_m_p_f']
])


def add_bh_correction_per_vignette(df):
    df = df[df.predictor != '(Intercept)'].copy()
    
    # Apply BH within each (model, prompt, vignette) group
    df['p_adj'] = np.nan
    
    # Notice we added 'vignette' to the groupby list!
    for (m, pr, v), grp in df.groupby(['model', 'prompt', 'vignette']):
        df.loc[grp.index, 'p_adj'] = false_discovery_control(grp['p_value'].values)
    
    return df

# Load, apply, and save
per_vignette_df = pd.read_csv("mixed_effects_all_vignettes_med_gemma.csv")
per_vig_bh = add_bh_correction_per_vignette(per_vignette_df)
per_vig_bh.to_csv("mixed_effects_all_vignettes_with_BH_med_gemma.csv", index=False)

# Verify what survives using the standard correction (p_adj_m_p)
print("=== Survives BH (all vignettes) ===")
print(per_vig_bh[per_vig_bh.p_adj < 0.05][
    ['model','prompt','vignette', 'predictor','Estimate','p_value','p_adj']
])


import numpy as np
var = pd.read_csv('all_variance_components_med_gemma.csv')
var['icc'] = var['vcov'] / (var['vcov'] + np.pi**2 / 3)
print(var[['model','prompt','vcov', 'sdcor', 'icc']])