import pandas as pd
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error


def build_model(input_dim):
    """
    Builds a simple feedforward neural network.
    This model is similar in spirit to the DeepPerf approach.
    """
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model


def main():
    # Define parameters
    systems = ['batlik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    num_repeats = 3  # Using 3 repeats for quick testing as it's heavy NN
    train_frac = 0.7  # Use 70% of data for training
    random_seed = 1  # Base seed for reproducibility

    for current_system in systems:
        datasets_location = f'datasets/{current_system}'
        csv_files = [f for f in os.listdir(datasets_location) if f.endswith('.csv')]

        for csv_file in csv_files:
            print(
                f'\n> System: {current_system}, Dataset: {csv_file}, Training fraction: {train_frac}, Repeats: {num_repeats}')
            data = pd.read_csv(os.path.join(datasets_location, csv_file))

            # Optionally, check for missing values, outliers, etc.

            metrics = {'MAPE': [], 'MAE': [], 'RMSE': []}
            for current_repeat in range(num_repeats):
                # Randomly split the dataset into training and testing sets
                train_data = data.sample(frac=train_frac, random_state=random_seed * current_repeat)
                test_data = data.drop(train_data.index)

                # Separate features and target
                training_X = train_data.iloc[:, :-1].values
                training_Y = train_data.iloc[:, -1].values
                testing_X = test_data.iloc[:, :-1].values
                testing_Y = test_data.iloc[:, -1].values

                # Scale the features using MinMaxScaler
                scaler = MinMaxScaler()
                training_X = scaler.fit_transform(training_X)
                testing_X = scaler.transform(testing_X)

                # Build and train the neural network model
                model = build_model(training_X.shape[1])
                model.fit(training_X, training_Y, epochs=100, batch_size=32, verbose=0)

                # Predict on the testing set
                predictions = model.predict(testing_X).flatten()

                # Compute evaluation metrics
                mape = mean_absolute_percentage_error(testing_Y, predictions)
                mae = mean_absolute_error(testing_Y, predictions)
                rmse = np.sqrt(mean_squared_error(testing_Y, predictions))

                metrics['MAPE'].append(mape)
                metrics['MAE'].append(mae)
                metrics['RMSE'].append(rmse)

            print('Average MAPE: {:.2f}'.format(np.mean(metrics['MAPE'])))
            print("Average MAE: {:.2f}".format(np.mean(metrics['MAE'])))
            print("Average RMSE: {:.2f}".format(np.mean(metrics['RMSE'])))


if __name__ == "__main__":
    main()
