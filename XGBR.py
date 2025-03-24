import pandas as pd
import os
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
import numpy as np


def main():
    """
    This function performs performance prediction using the XGBRegressor model.

    Systems: List of system names for which CSV datasets are available.
    num_repeats: Number of repetitions to mitigate stochastic variability.
    train_frac: Fraction of data used for training.
    random_seed: Base random seed for reproducibility.
    """
    # Specify parameters
    systems = ['batlik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    num_repeats = 30  # Number of repetitions (adjust as needed)
    train_frac = 0.7  # 70% training data
    random_seed = 1  # Base seed for reproducibility

    # Loop over each system's datasets
    for current_system in systems:
        datasets_location = f'datasets/{current_system}'  # Directory for the current system

        # List all CSV files in the directory
        csv_files = [f for f in os.listdir(datasets_location) if f.endswith('.csv')]

        for csv_file in csv_files:
            print(
                f'\n> System: {current_system}, Dataset: {csv_file}, Training fraction: {train_frac}, Repeats: {num_repeats}')

            # Load the dataset
            data = pd.read_csv(os.path.join(datasets_location, csv_file))

            # Initialize dictionary to store metrics for each repeat
            metrics = {'MAPE': [], 'MAE': [], 'RMSE': []}

            for current_repeat in range(num_repeats):
                # Randomly split the data into training and testing sets
                train_data = data.sample(frac=train_frac, random_state=random_seed * current_repeat)
                test_data = data.drop(train_data.index)

                # Split into features (X) and target (Y)
                training_X = train_data.iloc[:, :-1]  # All columns except last as features
                training_Y = train_data.iloc[:, -1]  # Last column as target performance value
                testing_X = test_data.iloc[:, :-1]
                testing_Y = test_data.iloc[:, -1]

                # Initialize the XGBRegressor model
                model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

                # Train the model on the training data
                model.fit(training_X, training_Y)

                # Predict performance values for the testing set
                predictions = model.predict(testing_X)

                # Calculate evaluation metrics for the current repeat
                mape = mean_absolute_percentage_error(testing_Y, predictions)
                mae = mean_absolute_error(testing_Y, predictions)
                rmse = np.sqrt(mean_squared_error(testing_Y, predictions))

                # Store the metrics
                metrics['MAPE'].append(mape)
                metrics['MAE'].append(mae)
                metrics['RMSE'].append(rmse)

            # Compute and display the average of the metrics across repeats
            print('Average MAPE: {:.2f}'.format(np.mean(metrics['MAPE'])))
            print("Average MAE: {:.2f}".format(np.mean(metrics['MAE'])))
            print("Average RMSE: {:.2f}".format(np.mean(metrics['RMSE'])))


if __name__ == "__main__":
    main()
