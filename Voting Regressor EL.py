import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
import numpy as np

def main():
    """
    This function performs performance prediction using an ensemble VotingRegressor.
    The ensemble combines Linear Regression, Random Forest, and XGBRegressor.
    For each dataset, the data is randomly split into 70% training and 30% testing.
    This process is repeated 30 times to minimize stochastic bias.
    """
    # Define parameters
    systems = ['batlik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    num_repeats = 30  # Repeat 30 times
    train_frac = 0.7  # 70% training data
    random_seed = 1   # Base seed for reproducibility

    # Initialize base models for the ensemble
    model_lr = LinearRegression()
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

    # Create an ensemble VotingRegressor that averages the predictions of the base models
    ensemble = VotingRegressor(estimators=[
        ('lr', model_lr),
        ('rf', model_rf),
        ('xgb', model_xgb)
    ])

    # Loop over each system's datasets
    for current_system in systems:
        datasets_location = f'datasets/{current_system}'  # Directory for current system
        csv_files = [f for f in os.listdir(datasets_location) if f.endswith('.csv')]

        for csv_file in csv_files:
            print(f'\n> System: {current_system}, Dataset: {csv_file}, Training fraction: {train_frac}, Repeats: {num_repeats}')
            data = pd.read_csv(os.path.join(datasets_location, csv_file))
            metrics = {'MAPE': [], 'MAE': [], 'RMSE': []}

            for current_repeat in range(num_repeats):
                # Randomly split the dataset
                train_data = data.sample(frac=train_frac, random_state=random_seed * current_repeat)
                test_data = data.drop(train_data.index)

                # Split features and target (assumes last column is target)
                training_X = train_data.iloc[:, :-1]
                training_Y = train_data.iloc[:, -1]
                testing_X = test_data.iloc[:, :-1]
                testing_Y = test_data.iloc[:, -1]

                # Train the ensemble model on the training data
                ensemble.fit(training_X, training_Y)
                predictions = ensemble.predict(testing_X)

                # Compute metrics
                mape = mean_absolute_percentage_error(testing_Y, predictions)
                mae = mean_absolute_error(testing_Y, predictions)
                rmse = np.sqrt(mean_squared_error(testing_Y, predictions))

                metrics['MAPE'].append(mape)
                metrics['MAE'].append(mae)
                metrics['RMSE'].append(rmse)

            # Output average metrics over the repeats
            print('Average MAPE: {:.2f}'.format(np.mean(metrics['MAPE'])))
            print("Average MAE: {:.2f}".format(np.mean(metrics['MAE'])))
            print("Average RMSE: {:.2f}".format(np.mean(metrics['RMSE'])))

if __name__ == "__main__":
    main()
