"""
Bootstrapped pooled mixed-effects model, vignette as random effect.

Cluster bootstrap over vignettes: resample vignette IDs with replacement,
refit the pooled glmer model on each resample, and collect the Estimate
for each predictor to build percentile confidence intervals.

Start here: QWEN | baseline only (Table 2 effects). Run with N_BOOT = 200
first as a smoke test, then bump to 1000 for the real CIs once verified.
"""

import pandas as pd
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

lme4 = importr('lme4')
utils = importr('utils')
base = importr('base')

AGGREDATED_DATA = "paper_results/Master_Aggregated_Data_EE.csv"
RUN_DATA = "paper_results/Master_RunLevel_Data_EE.csv"

print("Loading run-level CSV...")
df = pd.read_csv(RUN_DATA)


def fit_pooled(df, model_name, prompt_name, exclude_vignettes=None):
    subset = df[(df.model == model_name) & (df.prompt == prompt_name)].copy()

    if exclude_vignettes:
        subset = subset[~subset.vignette_id.isin(exclude_vignettes)]

    if subset.empty:
        return None, None, False

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

        # lme4 records convergence problems on the fitted object itself rather than
        # raising something rpy2 surfaces as a Python warning, so check it directly.
        conv_messages = ro.r(
            'function(m) { msgs <- m@optinfo$conv$lme4$messages; '
            'if (is.null(msgs)) "" else paste(msgs, collapse="; ") }'
        )(fit)[0]
        converged = (conv_messages == "")

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

        return coef_df, var_df, converged

    except Exception as e:
        print(f"  -> ERROR fitting pooled {model_name} {prompt_name}: {e}")
        return None, None, False

    finally:
        # Critical Garbage Collection so R doesn't crash on these massive datasets
        ro.r('gc()')


def resample_vignettes(subset, sampled_ids):
    """Build a resampled dataframe from already-drawn vignette IDs. If a vignette
    is drawn more than once, each copy gets its own unique vignette_id so the
    (1 | vignette_id) random intercept treats the repeats as separate clusters,
    not one collapsed one."""
    pieces = []
    for i, vid in enumerate(sampled_ids):
        piece = subset[subset.vignette_id == vid].copy()
        piece['vignette_id'] = f"{vid}_{i}"
        pieces.append(piece)

    return pd.concat(pieces, ignore_index=True)


def verify_relabeling(subset, vignette_ids, seed=12345):
    """Sanity check (the #1 bug to rule out): draw one resample and confirm that
    when a vignette is sampled more than once, each copy ends up with a distinct
    vignette_id rather than collapsing into a single cluster."""
    rng = np.random.default_rng(seed)
    sampled_ids = rng.choice(vignette_ids, size=len(vignette_ids), replace=True)
    resampled = resample_vignettes(subset, sampled_ids)

    n_draws = len(sampled_ids)
    n_labels = resampled['vignette_id'].nunique()
    n_dupes = n_draws - len(set(sampled_ids))
    status = "OK" if n_labels == n_draws else "BUG: duplicate draws collapsed into one cluster!"

    print("Relabeling spot-check:")
    print(f"  Sampled vignette IDs : {list(sampled_ids)}")
    print(f"  Relabeled vignette_id: {sorted(resampled['vignette_id'].unique(), key=str)}")
    print(f"  -> {n_dupes} duplicate draw(s), {n_labels}/{n_draws} distinct vignette_id labels -- {status}")


def bootstrap_pooled(df, model_name, prompt_name, vignette_ids, n_boot, seed=0):
    rng = np.random.default_rng(seed)

    subset = df[(df.model == model_name) & (df.prompt == prompt_name) &
                (df.vignette_id.isin(vignette_ids))].copy()

    boot_estimates = []
    n_failed = 0
    n_nonconverged = 0

    for b in range(n_boot):
        print(f"Bootstrap {model_name.upper()} | Prompt: {prompt_name} | resample {b + 1}/{n_boot} ...")

        sampled_ids = rng.choice(vignette_ids, size=len(vignette_ids), replace=True)
        resampled = resample_vignettes(subset, sampled_ids)
        coef, var, converged = fit_pooled(resampled, model_name, prompt_name)

        if coef is None:
            n_failed += 1
            continue

        if not converged:
            n_nonconverged += 1
            continue

        coef = coef.copy()
        coef['boot_iter'] = b
        boot_estimates.append(coef[['boot_iter', 'model', 'prompt', 'predictor', 'Estimate']])

    n_ok = len(boot_estimates)
    fail_rate = (n_failed + n_nonconverged) / n_boot
    print(f"  -> of {n_boot} resamples, {n_ok} converged "
          f"({n_failed} failed to fit, {n_nonconverged} fit but did not converge)")
    if fail_rate > 0.20:
        print(f"  -> WARNING: {fail_rate:.0%} of resamples failed/did not converge (>20%). "
              f"CIs below are based on a biased subset of resamples -- treat with caution.")

    if not boot_estimates:
        return None

    return pd.concat(boot_estimates, ignore_index=True)


def summarize_bootstrap(boot_df):
    """Percentile bootstrap CIs per predictor."""
    rows = []
    for predictor, grp in boot_df.groupby('predictor'):
        est = grp['Estimate'].values
        rows.append({
            'model': grp['model'].iloc[0],
            'prompt': grp['prompt'].iloc[0],
            'predictor': predictor,
            'n_boot': len(est),
            'mean': est.mean(),
            'median': np.median(est),
            'ci_low': np.percentile(est, 2.5),
            'ci_high': np.percentile(est, 97.5),
        })
    return pd.DataFrame(rows)


TABLE2_SIGNIFICANT_PREDICTORS = ['race_ambiguity_z', 'luminance_z', 'raceL', 'raceW', 'genderM']


def check_excludes_zero(ci_df, predictors=TABLE2_SIGNIFICANT_PREDICTORS):
    """Does the bootstrap 95% CI exclude zero for each Table 2 significant predictor?"""
    print("\nTable 2 predictor check (does the 95% CI exclude zero?):")
    for p in predictors:
        row = ci_df[ci_df.predictor == p]
        if row.empty:
            print(f"  {p}: not found in bootstrap output")
            continue
        lo, hi = row.iloc[0][['ci_low', 'ci_high']]
        excludes_zero = (lo > 0) or (hi < 0)
        print(f"  {p}: CI = [{lo:.4f}, {hi:.4f}] -- {'excludes zero' if excludes_zero else 'includes zero'}")


def compare_to_original(ci_df, orig_coef_df):
    """Bootstrap center (mean/median) should land close to the original point
    estimate. A large gap here means the resampling scheme is biased/broken."""
    merged = ci_df.merge(
        orig_coef_df[['predictor', 'Estimate']].rename(columns={'Estimate': 'orig_estimate'}),
        on='predictor', how='left'
    )
    merged['diff_median_vs_orig'] = merged['median'] - merged['orig_estimate']

    print("\nBootstrap center vs. original (non-bootstrapped) point estimate:")
    print(merged[['predictor', 'orig_estimate', 'mean', 'median', 'diff_median_vs_orig',
                  'ci_low', 'ci_high']].round(4))
    return merged


## A. Qwen baseline only (Table 2) -- verify here before extending to other cells

qwen_without_extremes = [2, 3, 6, 7, 8, 9, 11, 12]

# Original (non-bootstrapped) fit, used below to sanity-check the bootstrap center
print("Fitting original (non-bootstrapped) pooled model: QWEN | Prompt: baseline ...")
qwen_baseline_orig_coef, qwen_baseline_orig_var, qwen_baseline_orig_converged = fit_pooled(
    df, 'qwen', 'baseline',
    exclude_vignettes=[v for v in range(1, 13) if v not in qwen_without_extremes]
)

qwen_baseline_subset = df[(df.model == 'qwen') & (df.prompt == 'baseline') &
                           (df.vignette_id.isin(qwen_without_extremes))].copy()
verify_relabeling(qwen_baseline_subset, qwen_without_extremes)

N_BOOT = 1000  # quick check first; bump to 1000 for the real CIs once this runs cleanly

print(f"\nBootstrapping pooled model: QWEN | Prompt: baseline | N_BOOT={N_BOOT} ...")
qwen_baseline_boot = bootstrap_pooled(
    df, 'qwen', 'baseline', qwen_without_extremes, n_boot=N_BOOT, seed=0
)

if qwen_baseline_boot is not None:
    qwen_baseline_boot.to_csv(f"qwen_baseline_bootstrap_estimates_n{N_BOOT}.csv", index=False)

    qwen_baseline_ci = summarize_bootstrap(qwen_baseline_boot)
    qwen_baseline_ci.to_csv(f"qwen_baseline_bootstrap_ci_n{N_BOOT}.csv", index=False)

    print("\n" + "=" * 80)
    print(qwen_baseline_ci.round(4))
    print("=" * 80)

    if qwen_baseline_orig_coef is not None:
        compare_to_original(qwen_baseline_ci, qwen_baseline_orig_coef)

    check_excludes_zero(qwen_baseline_ci)
else:
    print("No successful resamples -- nothing to save.")
