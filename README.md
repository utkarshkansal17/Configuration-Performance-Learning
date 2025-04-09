# Configuration Performance Learning

## 🔍 Objective

To evaluate how effectively different regression models can predict performance outcomes (like execution time or compression ratio) across highly configurable software systems.

## 📊 Models Implemented

- Linear Regression (Baseline)
- Neural Network
- Random Forest
- Support Vector Regression (SVR)
- SVR with Hyperparameter Tuning
- Voting Regressor Ensemble
- Voting Regressor with Tuning
- XGBoost Regressor


## 🗂️ Project Structure

```
Configuration-Performance-Learning/
├── Comparison by System.py                # Generates heatmaps to compare models across systems
├── NN.py                                  # Implements and trains the Neural Network model
├── README.md                              # Project documentation
├── Results/                               # Contains outputs from experiments
│   ├── Comparisons/                       # Consolidated results and system-level visualizations
│   │   ├── RES_system2_comparison_average_MAE.csv         # System-wise MAE comparison
│   │   ├── RES_system2_comparison_average_MAPE.csv        # System-wise MAPE comparison
│   │   ├── RES_system2_comparison_average_RMSE.csv        # System-wise RMSE comparison
│   │   ├── RES_system_heatmap2_average_MAE_clean.png      # MAE heatmap
│   │   ├── RES_system_heatmap2_average_MAPE_clean.png     # MAPE heatmap
│   │   └── RES_system_heatmap2_average_RMSE_clean.png     # RMSE heatmap
│   ├── Linear Regression Baseline.csv     # Raw results of linear regression baseline
│   ├── NN.csv                              # Neural network performance per system
│   ├── RandomForest.csv                    # Random forest model results
│   ├── SVM with Tunning.csv                # SVM model with hyperparameter tuning
│   ├── SVR.csv                             # Standard Support Vector Regression results
│   ├── Stat_test.ipynb                     # T-test statistical comparison of all models
│   ├── Voting Regressor EL with Tunning.csv   # Ensemble voting regressor with tuning
│   ├── Voting Regressor EL.csv             # Ensemble voting regressor without tuning
│   └── XGBR.csv                             # XGBoost model predictions
├── SVM with GridSearch tunning.py         # SVM with GridSearchCV for best hyperparameters
├── SVM with Tunning.py                    # SVM with manual tuning
├── SVM.py                                 # Basic SVM model
├── Voting Regressor EL.py                 # Implements ensemble learning using voting regressor
├── Voting Regressor with Tunning.py       # Voting regressor with feature engineering and tuning
├── XGBR.py                                # XGBoost model script
├── comparisons.py                         # Compares models and generates average results
├── datasets/                              # Input datasets for each system
│   ├── Irzip/                              # Long-range compression datasets
│   ├── batlik/                             # Image processing datasets
│   ├── dconvert/                           # Document conversion test cases
│   ├── h2/                                 # Database system configuration datasets
│   ├── jump3r/                             # Audio encoding workloads
│   ├── kanzi/                              # General compression tool configurations
│   ├── x264/                               # Video encoding benchmark data
│   ├── xz/                                 # Compression datasets for xz utility
│   └── z3/                                 # SMT theorem prover configuration performance data
├── lab2_solution.py                       # Central orchestration for all model training and evaluation
└── randomforest.py                        # Script for training and evaluating Random Forest

```
## ⚙️ Installation & Usage

### 🔧 Prerequisites

Make sure you have the following installed:

- Python 3.8 or higher
- pip (Python package manager)
### 🛠️ Required Libraries

To run the project, install the following Python packages:

- **Data Handling:** `numpy`, `pandas`
- **Machine Learning Models:** `scikit-learn`, `xgboost`, `tensorflow`, `keras`, `skorch`
- **Statistical Analysis:** `scipy`, `statsmodels` *(optional for significance testing)*
- **Visualization:** `matplotlib`, `seaborn`, `altair`
- **Notebooks Support:** `jupyterlab`, `notebook`, `ipykernel`, `nbformat`, `nbconvert`
- **Experiment Tracking:** `wandb`
- **Utilities:** `joblib`, `tqdm`, `tabulate`
- **PyTorch Suite:** `torch`, `torchvision`, `torchaudio`, `torchtext`, `torchdata`

You can install all at once using:

```bash
pip install -r requirements.txt
```

## 🚀 Running the Code

Run any model script:

python XGBR.py  # notedown the output

python "Voting Regressor with Tunning.py" # notedown the output

Generate system-wise comparison heatmaps:

python "Comparison by System.py"

After running the scripts the output will look this

```bash
> System: batlik, Dataset: corona.csv, Training fraction: 0.7, Repeats: 3
Average MAPE: 0.02
Average MAE: 0.04
Average RMSE: 0.05
```


Perform statistical significance tests (via Jupyter Notebook):

jupyter notebook "Results/Stat_test.ipynb"

## 📝 Logging Results to CSV

To standardize the evaluation output across all model scripts, use a structured list called `results` to collect and store performance metrics for each dataset within a system.

### 🔹 Step-by-Step Logging Guide

1. **Initialize a results list** at the start of your script (before the main loop):

```python
results = []
```
2. After computing average metrics inside your inner loop (i.e., after printing the final scores), log the result for that dataset like this:

```python

# Collect results for CSV export
results.append({ 
    'System': current_system,
    'Dataset': csv_file,
    'TrainFraction': train_frac,
    'Repeats': num_repeats,
    'Average MAPE': avg_mape,
    'Average MAE': avg_mae,
    'Average RMSE': avg_rmse
})
```
At the end of the script, save all results to a CSV file in the Results/ directory:

```python
import pandas as pd
os.makedirs('Results', exist_ok=True)
pd.DataFrame(results).to_csv('Results/YourModelName.csv', index=False)
```
📌 Replace "YourModelName.csv" with the specific filename for your model (e.g., XGBR.csv, Voting Regressor EL with Tunning.csv, etc.).

| System   | Dataset         | TrainFraction | Repeats | Average MAPE | Average MAE | Average RMSE |
|----------|-----------------|----------------|---------|----------------|---------------|----------------|
| batlik   | corona.csv      | 0.7            | 30      | 0.023          | 0.045         | 0.067          |
| dconvert | jpeg-small.csv  | 0.7            | 30      | 0.012          | 0.034         | 0.058          |
| ...      | ...             | ...            | ...     | ...            | ...           | ...            |

You can also check the Results folder csv files for more understanding of the above output results

This logging mechanism ensures reproducibility and simplifies comparison across models and systems. It also enables downstream statistical analysis and visualization (e.g., heatmaps, t-tests).
