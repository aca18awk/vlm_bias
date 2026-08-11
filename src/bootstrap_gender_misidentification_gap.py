"""
Bootstrap CI on the female vs. male gender-misidentification gap.

Independent units are FACES (one row per photo_id), not individual mentions.
Each face is resampled with replacement within its true_gender group, counts
are pooled across the resample, and the mismatch rate is computed from the
pooled counts (not the mean of per-face rates) so faces are weighted by how
many explicit gender mentions they have.
"""

import re

import numpy as np
import pandas as pd

RUN_DATA = "Master_RunLevel_Data_Medgemma_EE.csv"
N_BOOT = 2000
SEED = 0

MALE_PATTERN = re.compile(r'\b(male|males|he|him|his|man|men|prostate|testicular)\b')
FEMALE_PATTERN = re.compile(r'\b(female|females|she|her|hers|woman|women|gynecological|gynecology|pregnant)\b')


def build_per_face(filepath=RUN_DATA):
    """One row per face: race, photo_id, true_gender, total_mentions, mismatches.

    total_mentions/mismatches only count explicit male/female pronoun hits
    (ambiguous mentions -- e.g. just 'sex'/'gender'/'30'/'old' -- are excluded,
    since they can't be judged as a match or a mismatch)."""
    df = pd.read_csv(filepath)
    df_gender = df[df['mentions_gender'] == 1].copy()

    df_gender['true_gender'] = df_gender['photo_id'].str[1].map({'M': 'Male', 'F': 'Female'})
    df_gender['race'] = df_gender['photo_id'].str[0]
    df_gender = df_gender.dropna(subset=['true_gender'])

    def check_match(row):
        evidence = str(row['evidence_gender']).lower()
        has_male = bool(MALE_PATTERN.search(evidence))
        has_female = bool(FEMALE_PATTERN.search(evidence))

        if has_male and not has_female:
            return 'Match' if row['true_gender'] == 'Male' else 'Mismatch'
        elif has_female and not has_male:
            return 'Match' if row['true_gender'] == 'Female' else 'Mismatch'
        else:
            return 'Other'

    df_gender['match_status'] = df_gender.apply(check_match, axis=1)

    face = df_gender.groupby(['race', 'photo_id', 'true_gender', 'match_status']).size().unstack(fill_value=0)
    face['total_mentions'] = face.get('Match', 0) + face.get('Mismatch', 0)
    face['mismatches'] = face.get('Mismatch', 0)
    face['mismatch_pct'] = (face['mismatches'] / face['total_mentions'].replace(0, float('nan')) * 100).round(2)

    return face[['total_mentions', 'mismatches', 'mismatch_pct']].reset_index()


def bootstrap_female_male_gap(per_face, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)

    female = per_face[per_face.true_gender == 'Female'][['mismatches', 'total_mentions']].values
    male = per_face[per_face.true_gender == 'Male'][['mismatches', 'total_mentions']].values

    def pooled_rate(arr):
        return 100 * arr[:, 0].sum() / arr[:, 1].sum()

    # point estimates
    f_point = pooled_rate(female)
    m_point = pooled_rate(male)

    f_boot, m_boot, gap_boot = [], [], []
    for _ in range(n_boot):
        # resample FACES with replacement, within each group
        f_idx = rng.integers(0, len(female), size=len(female))
        m_idx = rng.integers(0, len(male), size=len(male))
        f_rate = pooled_rate(female[f_idx])
        m_rate = pooled_rate(male[m_idx])
        f_boot.append(f_rate)
        m_boot.append(m_rate)
        gap_boot.append(f_rate - m_rate)

    def ci(a):
        return (np.percentile(a, 2.5), np.percentile(a, 97.5))

    return {
        'female_rate': round(f_point, 2), 'female_CI': tuple(round(x, 2) for x in ci(f_boot)),
        'male_rate': round(m_point, 2), 'male_CI': tuple(round(x, 2) for x in ci(m_boot)),
        'gap': round(f_point - m_point, 2), 'gap_CI': tuple(round(x, 2) for x in ci(gap_boot)),
    }


if __name__ == "__main__":
    print(f"Loading {RUN_DATA}...")
    per_face = build_per_face(RUN_DATA)
    per_face.to_csv("per_face_gender_mismatch.csv", index=False)

    n_female = (per_face.true_gender == 'Female').sum()
    n_male = (per_face.true_gender == 'Male').sum()
    print(f"Faces: {n_female} female, {n_male} male")

    result = bootstrap_female_male_gap(per_face, n_boot=N_BOOT, seed=SEED)

    print(f"\nFemale rate: {result['female_rate']:.2f}% "
          f"(95% CI [{result['female_CI'][0]:.2f}, {result['female_CI'][1]:.2f}])")
    print(f"Male rate:   {result['male_rate']:.2f}% "
          f"(95% CI [{result['male_CI'][0]:.2f}, {result['male_CI'][1]:.2f}])")
    print(f"Gap (F-M):   {result['gap']:.2f}pp "
          f"(95% CI [{result['gap_CI'][0]:.2f}, {result['gap_CI'][1]:.2f}])")
