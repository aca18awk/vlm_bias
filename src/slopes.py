

import pandas as pd
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr 

AGGREDATED_DATA = "paper_results/Master_Aggregated_Data_EE.csv"
RUN_DATA = "paper_results/Master_RunLevel_Data_EE.csv"

print("Loading run-level CSV...")
df = pd.read_csv(RUN_DATA)

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
        "luminance_z + race + gender + (1 + race_ambiguity_z | vignette_id)"
    )
    
    try:
        with localconverter(ro.default_converter + numpy2ri.converter + pandas2ri.converter):
            r_df = ro.conversion.py2rpy(subset)
            
        fit = lme4.glmer(formula_r, data=r_df, family=ro.r('binomial()'))

        conv_messages = ro.r(
            'function(m) { msgs <- m@optinfo$conv$lme4$messages; '
            'if (is.null(msgs)) "" else paste(msgs, collapse="; ") }'
        )(fit)[0]
        print(f"  Convergence messages: {conv_messages if conv_messages else 'CONVERGED CLEANLY'}")
        
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

qwen_without_extremes = [2,3,6,7,8,9,11,12]

# for m in ['qwen', 'llama']:
for m in ['qwen']:
    excl = [v for v in range(1, 13) if v not in qwen_without_extremes] if m=='qwen' else None
    # for p in ['baseline', 'ignore', 'acknowledge']:
    for p in ['baseline']:

        print(f"Fitting pooled model: {m.upper()} | Prompt: {p} ...")
        coef, var = fit_pooled(df, m, p, exclude_vignettes=excl)
        
        if coef is not None: 
            results_pooled.append(coef)
        if var is not None: 
            variance_components.append(var)

# Combine and save
pooled_df = pd.concat(results_pooled, ignore_index=True)
var_df_all = pd.concat(variance_components, ignore_index=True)

print(var_df_all)

# pooled_df.to_csv("all_pooled_results_slopes.csv", index=False)
var_df_all.to_csv("all_variance_components_slopes.csv", index=False)

print("\n" + "="*80)
print("SUCCESS! Pooled models completed and saved.")
print("="*80)