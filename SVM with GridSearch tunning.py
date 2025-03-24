import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error

def main():
    # Define parameters
    systems = ['batlik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    num_repeats = 10
    train_frac = 0.7
    random_seed = 1

    # Define the SVR model as the final regressor
    svr_model = SVR()

    # Specify categorical column indices (update these indices as needed)
    categorical_cols = [0, 2]

    # Build a preprocessor using ColumnTransformer:
    # Apply OneHotEncoder to categorical columns and pass through the rest.
    preprocessor = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols)
    ], remainder='passthrough')

    # Create the pipeline: preprocessing -> polynomial features -> scaling -> SVR
    pipeline = Pipeline([
        ('preproc', preprocessor),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler()),
        ('svr', svr_model)
    ])

    # Set up a parameter grid for hyperparameter tuning
    param_grid = {
        'poly__degree': [1, 2],
        'svr__C': [0.1, 1, 10],
        'svr__epsilon': [0.01, 0.1, 0.5],
        'svr__kernel': ['rbf', 'linear']
    }

    # Loop over each system and dataset
    for current_system in systems:
        datasets_location = f'datasets/{current_system}'
        csv_files = [f for f in os.listdir(datasets_location) if f.endswith('.csv')]
        for csv_file in csv_files:
            print(f'\n> System: {current_system}, Dataset: {csv_file}, Training fraction: {train_frac}, Repeats: {num_repeats}')
            data = pd.read_csv(os.path.join(datasets_location, csv_file))
            metrics = {'MAPE': [], 'MAE': [], 'RMSE': []}

            for current_repeat in range(num_repeats):
                # Split data into training and testing sets
                train_data = data.sample(frac=train_frac, random_state=random_seed * current_repeat)
                test_data = data.drop(train_data.index)

                X_train = train_data.iloc[:, :-1].values
                y_train = train_data.iloc[:, -1].values
                X_test = test_data.iloc[:, :-1].values
                y_test = test_data.iloc[:, -1].values

                # Use GridSearchCV for hyperparameter tuning with a 3-fold CV
                grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_

                predictions = best_model.predict(X_test)
                mape = mean_absolute_percentage_error(y_test, predictions)
                mae = mean_absolute_error(y_test, predictions)
                rmse = np.sqrt(mean_squared_error(y_test, predictions))

                metrics['MAPE'].append(mape)
                metrics['MAE'].append(mae)
                metrics['RMSE'].append(rmse)

            # Output the average metrics for the current dataset
            print('Average MAPE: {:.2f}'.format(np.mean(metrics['MAPE'])))
            print("Average MAE: {:.2f}".format(np.mean(metrics['MAE'])))
            print("Average RMSE: {:.2f}".format(np.mean(metrics['RMSE'])))

if __name__ == "__main__":
    main()
