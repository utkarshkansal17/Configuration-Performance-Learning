import os
import pandas as pd
import matplotlib.pyplot as plt

# Folder where your CSVs are located
results_folder = 'results'

# Store metrics here
metrics = {}

for filename in os.listdir(results_folder):
    if filename.endswith('.csv'):
        filepath = os.path.join(results_folder, filename)
        model_name = filename.replace('.csv', '')

        df = pd.read_csv(filepath)

        # Check and calculate mean of available metrics
        mae_mean = df['Average MAE'].mean() if 'Average MAE' in df.columns else None
        rmse_mean = df['Average RMSE'].mean() if 'Average RMSE' in df.columns else None

        metrics[model_name] = {
            'Average MAE': round(mae_mean, 4) if mae_mean is not None else None,
            'Average RMSE': round(rmse_mean, 4) if rmse_mean is not None else None
        }

# Convert to DataFrame
results_df = pd.DataFrame.from_dict(metrics, orient='index')
results_df = results_df.sort_values(by='Average RMSE')
print("✅ Model Comparison Table:\n")
print(results_df)

# Save table to CSV
results_df.to_csv('model_comparison_summary.csv')

# Plot MAE
if not results_df['Average MAE'].isnull().all():
    plt.figure(figsize=(12, 5))
    plt.bar(results_df.index, results_df['Average MAE'], alpha=0.7)
    plt.title('Average MAE Comparison Across Models')
    plt.xticks(rotation=45)
    plt.ylabel('Average MAE')
    plt.tight_layout()
    plt.savefig('average_mae_comparison.png')
    plt.show()
else:
    print("⚠️ No data found for Average MAE.")

# Plot RMSE
if not results_df['Average RMSE'].isnull().all():
    plt.figure(figsize=(12, 5))
    plt.bar(results_df.index, results_df['Average RMSE'], color='orange', alpha=0.7)
    plt.title('Average RMSE Comparison Across Models')
    plt.xticks(rotation=45)
    plt.ylabel('Average RMSE')
    plt.tight_layout()
    plt.savefig('average_rmse_comparison.png')
    plt.show()
else:
    print("⚠️ No data found for Average RMSE.")
