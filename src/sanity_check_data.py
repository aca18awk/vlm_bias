import pandas as pd

CSV_PATH = "Master_Aggregated_Data.csv"

print("Loading run-level CSV...")
df = pd.read_csv(CSV_PATH)

print("\n── Diagnostics ──")
# print(f"Shape: {df_aggregated.shape}")

# incomplete = df_aggregated[df_aggregated['n_total'] < 20]
# print(f"Cells with < 20 runs: {len(incomplete)}" + (" ← WARNING" if len(incomplete) else " ✓"))
# if len(incomplete):
#     print(incomplete[['model', 'prompt', 'photo_id', 'vignette_id', 'n_total']].to_string())

# print(f"\nn_total: min={df_aggregated['n_total'].min()}  max={df_aggregated['n_total'].max()}  mean={df_aggregated['n_total'].mean():.1f}")

# print("\nHead:")
# print(df_aggregated.head().to_string())
# print("\nCell sizes (model × prompt × race × gender):")
# print(df_aggregated.groupby(['model', 'prompt', 'race', 'gender']).size().to_string())

# Should match your earlier V11 Qwen baseline by-race CE
# qwen_v11 = df[(df.model=='qwen') & (df.vignette_id==11) & (df.prompt=='baseline')]
# print(qwen_v11.groupby('race')['admit_rate'].mean())


# print(df.groupby(['model','vignette_id'])['vignette_class'].first())

# print(df[df.n_total<20].groupby(['model','prompt','race','gender']).size())

# print(df[df.n_total==50]['photo_id'].unique())

# Reproduce per-vignette mean admit rate across all faces
for model in ['qwen', 'llama']:
    for vid in [3, 11]:  # Check 2 borderline vignettes
        rate = df[(df.model==model) & (df.vignette_id==vid) & 
                  (df.prompt=='baseline')]['admit_rate'].mean()
        print(f"{model} V{vid:02d}: {rate:.2f}")


print(df[df.photo_id != 'no_photo'].groupby('photo_id').size().describe())

print(df[df.photo_id != 'no_photo'].groupby(['race','gender'])['photo_id'].nunique())