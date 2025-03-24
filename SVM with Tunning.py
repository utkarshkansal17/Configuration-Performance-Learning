import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error

def main():
    # Define parameters
    systems = ['batlik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    num_repeats = 30  # Reduced repeats for faster processing
    train_frac = 0.7
    random_seed = 1

    # Specify categorical column indices (update these indices as needed)
    categorical_cols = [0, 2]

    # Build the preprocessor using ColumnTransformer:
    # OneHotEncoder for categorical columns and pass through the rest.
    preprocessor = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols)
    ], remainder='passthrough')

    # Create the pipeline: preprocessing -> polynomial features -> scaling -> SVR
    pipeline = Pipeline([
        ('preproc', preprocessor),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler()),
        ('svr', SVR())
    ])

    # Define a parameter distribution for RandomizedSearchCV for faster tuning
    param_dist = {
        'poly__degree': [1, 2],
        'svr__C': [0.1, 1, 10],
        'svr__epsilon': [0.01, 0.1, 0.5],
        'svr__kernel': ['rbf', 'linear'],
        'svr__gamma': ['scale', 'auto']
    }

    # Use RandomizedSearchCV with fewer iterations and 2-fold CV for speed
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        cv=2,
        scoring='neg_mean_squared_error',
        n_iter=5,         # Reduced number of iterations
        n_jobs=-1,
        random_state=42
    )

    # Loop over each system's datasets
    for current_system in systems:
        datasets_location = f'datasets/{current_system}'
        csv_files = [f for f in os.listdir(datasets_location) if f.endswith('.csv')]
        for csv_file in csv_files:
            print(f'\n> System: {current_system}, Dataset: {csv_file}, Training fraction: {train_frac}, Repeats: {num_repeats}')
            data = pd.read_csv(os.path.join(datasets_location, csv_file))
            metrics = {'MAPE': [], 'MAE': [], 'RMSE': []}

            for current_repeat in range(num_repeats):
                # Randomly split the dataset
                train_data = data.sample(frac=train_frac, random_state=random_seed * current_repeat)
                test_data = data.drop(train_data.index)

                X_train = train_data.iloc[:, :-1].values
                y_train = train_data.iloc[:, -1].values
                X_test = test_data.iloc[:, :-1].values
                y_test = test_data.iloc[:, -1].values

                # Fit the model using RandomizedSearchCV for hyperparameter tuning
                random_search.fit(X_train, y_train)
                best_model = random_search.best_estimator_
                predictions = best_model.predict(X_test)

                # Compute evaluation metrics
                mape = mean_absolute_percentage_error(y_test, predictions)
                mae = mean_absolute_error(y_test, predictions)
                rmse = np.sqrt(mean_squared_error(y_test, predictions))

                metrics['MAPE'].append(mape)
                metrics['MAE'].append(mae)
                metrics['RMSE'].append(rmse)

            # Print the average metrics over all repeats
            print('Average MAPE: {:.2f}'.format(np.mean(metrics['MAPE'])))
            print("Average MAE: {:.2f}".format(np.mean(metrics['MAE'])))
            print("Average RMSE: {:.2f}".format(np.mean(metrics['RMSE'])))

if __name__ == "__main__":
    main()
