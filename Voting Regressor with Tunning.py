import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error


def main():
    systems = ['batlik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    num_repeats = 30
    train_frac = 0.7
    random_seed = 1

    ensemble = VotingRegressor(estimators=[
        ('lr', LinearRegression()),
        ('rf', RandomForestRegressor(random_state=42)),
        ('xgb', XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
    ])

    # Specify categorical columns (update indices as needed)
    categorical_cols = [0, 2]

    # Use sparse_output=False for OneHotEncoder (compatible with scikit-learn 1.6.1)
    preprocessor = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols)
    ], remainder='passthrough')

    pipeline = Pipeline([
        ('preproc', preprocessor),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler()),
        ('voting', ensemble)
    ])

    param_grid = {
        'poly__degree': [1, 2],
        'voting__rf__n_estimators': [50, 100],
        'voting__rf__max_depth': [None, 10],
        'voting__xgb__n_estimators': [50, 100],
        'voting__xgb__learning_rate': [0.1, 0.01]
    }

    for current_system in systems:
        datasets_location = f'datasets/{current_system}'
        csv_files = [f for f in os.listdir(datasets_location) if f.endswith('.csv')]
        for csv_file in csv_files:
            print(
                f'\n> System: {current_system}, Dataset: {csv_file}, Training fraction: {train_frac}, Repeats: {num_repeats}')
            data = pd.read_csv(os.path.join(datasets_location, csv_file))
            metrics = {'MAPE': [], 'MAE': [], 'RMSE': []}

            for current_repeat in range(num_repeats):
                train_data = data.sample(frac=train_frac, random_state=random_seed * current_repeat)
                test_data = data.drop(train_data.index)

                X_train = train_data.iloc[:, :-1].values
                y_train = train_data.iloc[:, -1].values
                X_test = test_data.iloc[:, :-1].values
                y_test = test_data.iloc[:, -1].values

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

            print('Average MAPE: {:.2f}'.format(np.mean(metrics['MAPE'])))
            print("Average MAE: {:.2f}".format(np.mean(metrics['MAE'])))
            print("Average RMSE: {:.2f}".format(np.mean(metrics['RMSE'])))


if __name__ == "__main__":
    main()
