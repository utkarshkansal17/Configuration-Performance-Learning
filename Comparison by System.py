import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Folder containing your CSV files
results_folder = 'results'

# The metric you want to extract (as it appears in original file)
target_metric_original = 'Average MAPE'
#target_metric_original = 'Average RMSE'
#target_metric_original = 'Average MAE'

# Normalized column name we'll match after lowercasing and stripping
target_metric_normalized = target_metric_original.strip().lower()

# Dictionary: {system: {model: avg_metric}}
system_data = {}

for filename in os.listdir(results_folder):
    if filename.endswith('.csv'):
        filepath = os.path.join(results_folder, filename)
        model_name = filename.replace('.csv', '')

        try:
            df = pd.read_csv(filepath)

            # Normalize column names
            df.columns = [col.strip().lower() for col in df.columns]

            if target_metric_normalized not in df.columns or 'system' not in df.columns:
                print(f"⚠️ Skipping {filename} — required column missing.")
                continue

            # Group by system and calculate mean metric
            system_means = df.groupby('system')[target_metric_normalized].mean()

            for system, value in system_means.items():
                if system not in system_data:
                    system_data[system] = {}
                system_data[system][model_name] = round(value, 4)

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

# Convert to DataFrame
system_df = pd.DataFrame.from_dict(system_data, orient='index')
system_df = system_df.sort_index()

# Save output
output_csv = f'system2_comparison_{target_metric_normalized.replace(" ", "_")}.csv'
system_df.to_csv(output_csv)
print(f"\n✅ Saved system-level comparison table to: {output_csv}")

# Plot heatmap
if not system_df.empty:
    plt.figure(figsize=(16, 8))  # Wider and taller figure
    ax = sns.heatmap(
        system_df,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        linewidths=0.5,
        annot_kws={"size": 8},
        cbar_kws={'label': target_metric_original},
        vmax=np.nanpercentile(system_df.values, 95)  # Cap extreme outliers for better contrast
    )

    # Titles and labels
    plt.title(f'{target_metric_original} per System per Model', fontsize=14)
    plt.ylabel('System', fontsize=12)
    plt.xlabel('Model', fontsize=12)

    # Ticks
    plt.xticks(rotation=90, ha='center', fontsize=10)
    plt.yticks(fontsize=10)

    # Save and show
    output_png = f'system_heatmap2_{target_metric_normalized.replace(" ", "_")}_clean.png'
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.show()
    print(f"✅ Heatmap saved to: {output_png}")
else:
    print("⚠️ No valid data found to plot.")
