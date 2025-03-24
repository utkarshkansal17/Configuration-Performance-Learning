import pandas as pd
import os
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
import numpy as np


def main():
    """
    This function performs performance prediction using Support Vector Regression (SVR).

    For each dataset, the data is randomly split into 70% training and 30% testing.
    This process is repeated 30 times to minimize stochastic bias.
    """
    # Define parameters
    systems = ['batlik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    num_repeats = 30  # Number of repetitions
    train_frac = 0.7  # 70% of the data for training
    random_seed = 1  # Base seed for reproducibility

    # Loop over each system's datasets
    for current_system in systems:
        datasets_location = f'datasets/{current_system}'  # Directory path for current system

        # List all CSV files in the current system's directory
        csv_files = [f for f in os.listdir(datasets_location) if f.endswith('.csv')]

        for csv_file in csv_files:
            print(
                f'\n> System: {current_system}, Dataset: {csv_file}, Training fraction: {train_frac}, Repeats: {num_repeats}')
            data = pd.read_csv(os.path.join(datasets_location, csv_file))

            # Dictionary to store evaluation metrics for each repeat
            metrics = {'MAPE': [], 'MAE': [], 'RMSE': []}

            for current_repeat in range(num_repeats):
                # Randomly split data into training and testing sets
                train_data = data.sample(frac=train_frac, random_state=random_seed * current_repeat)
                test_data = data.drop(train_data.index)

                # Split the data into features (X) and target (Y)
                training_X = train_data.iloc[:, :-1]  # All columns except the last one as features
                training_Y = train_data.iloc[:, -1]  # Last column is the target performance value
                testing_X = test_data.iloc[:, :-1]
                testing_Y = test_data.iloc[:, -1]

                # Initialize the SVR model; using the RBF kernel is a common choice for non-linear regression
                model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
                model.fit(training_X, training_Y)

                # Predict performance values on the testing set
                predictions = model.predict(testing_X)

                # Calculate evaluation metrics for the current repeat
                mape = mean_absolute_percentage_error(testing_Y, predictions)
                mae = mean_absolute_error(testing_Y, predictions)
                rmse = np.sqrt(mean_squared_error(testing_Y, predictions))

                # Store the metrics
                metrics['MAPE'].append(mape)
                metrics['MAE'].append(mae)
                metrics['RMSE'].append(rmse)

            # Print the average metrics over all repeats
            print('Average MAPE: {:.2f}'.format(np.mean(metrics['MAPE'])))
            print("Average MAE: {:.2f}".format(np.mean(metrics['MAE'])))
            print("Average RMSE: {:.2f}".format(np.mean(metrics['RMSE'])))


if __name__ == "__main__":
    main()
