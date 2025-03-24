import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
import numpy as np


def main():
    """
    This function performs performance prediction using the Random Forest Regressor.

    Parameters:
    - systems: List of system names, where each system has its datasets in CSV format.
    - num_repeats: Number of repetitions to mitigate stochastic variability.
    - train_frac: Fraction of the data used for training.
    - random_seed: Base random seed for reproducibility.
    """
    # Define parameters
    systems = ['batlik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    num_repeats = 30  # Number of repetitions
    train_frac = 0.7  # 70% of the data for training
    random_seed = 1  # Base seed for reproducibility

    # Loop over each system's datasets
    for current_system in systems:
        datasets_location = f'datasets/{current_system}'  # Construct path to dataset folder

        # List all CSV files in the current system's directory
        csv_files = [f for f in os.listdir(datasets_location) if f.endswith('.csv')]

        for csv_file in csv_files:
            print(
                f'\n> System: {current_system}, Dataset: {csv_file}, Training data fraction: {train_frac}, Number of repeats: {num_repeats}')
            data = pd.read_csv(os.path.join(datasets_location, csv_file))

            # Dictionary to store metrics for each repeat
            metrics = {'MAPE': [], 'MAE': [], 'RMSE': []}

            for current_repeat in range(num_repeats):
                # Randomly split data into training and testing sets
                train_data = data.sample(frac=train_frac, random_state=random_seed * current_repeat)
                test_data = data.drop(train_data.index)

                # Split the data into features (X) and target (Y)
                training_X = train_data.iloc[:, :-1]  # All columns except the last one
                training_Y = train_data.iloc[:, -1]  # Last column is the target performance value
                testing_X = test_data.iloc[:, :-1]
                testing_Y = test_data.iloc[:, -1]

                # Initialize and train the Random Forest Regressor
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(training_X, training_Y)

                # Predict on the testing set
                predictions = model.predict(testing_X)

                # Calculate evaluation metrics for the current repeat
                mape = mean_absolute_percentage_error(testing_Y, predictions)
                mae = mean_absolute_error(testing_Y, predictions)
                rmse = np.sqrt(mean_squared_error(testing_Y, predictions))

                # Append the metrics
                metrics['MAPE'].append(mape)
                metrics['MAE'].append(mae)
                metrics['RMSE'].append(rmse)

            # Output average metrics over the repeats
            print('Average MAPE: {:.2f}'.format(np.mean(metrics['MAPE'])))
            print("Average MAE: {:.2f}".format(np.mean(metrics['MAE'])))
            print("Average RMSE: {:.2f}".format(np.mean(metrics['RMSE'])))


if __name__ == "__main__":
    main()
