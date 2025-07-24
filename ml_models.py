from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


def linear_regression_modeling(train_df, test_df):
    """
    Trains a Linear Regression model on train_df and evaluates on test_df.
    Applies StandardScaler to numeric features and plots predicted vs. actual values,
    as well as a horizontal bar plot of regression coefficients.
    """
    # Separate features and target from training and testing data
    X_train = train_df.drop('firmness', axis=1)
    y_train = train_df['firmness']
    X_test = test_df.drop('firmness', axis=1)
    y_test = test_df['firmness']

    # Select numeric columns for scaling
    X_train_numeric = X_train.select_dtypes(include=[np.number])
    X_test_numeric = X_test.select_dtypes(include=[np.number])

    # Scale the numeric features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_numeric)
    X_test_scaled = scaler.transform(X_test_numeric)

    # Fit the Linear Regression model
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)

    # Predict on the test set
    y_pred = lr.predict(X_test_scaled)

    # Evaluation metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"Linear Regression RMSE: {rmse:.4f}")
    print(f"Linear Regression R²: {r2:.4f}")

    # Plot predicted vs actual values
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', label='Ideal')
    plt.xlabel("Actual Firmness")
    plt.ylabel("Predicted Firmness")
    plt.title("Linear Regression: Predicted vs Actual Firmness")
    plt.legend()
    plt.show()

    # Coefficient Plot: extract feature names and coefficients from the model
    feature_names = X_train_numeric.columns
    coefficients = lr.coef_

    # Create a DataFrame for easier plotting and sort by absolute coefficient value
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    })
    coef_df['AbsCoefficient'] = coef_df['Coefficient'].abs()
    coef_df = coef_df.sort_values(by='AbsCoefficient', ascending=False)

    # Plot a horizontal bar chart of coefficients
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Coefficient', y='Feature', data=coef_df, orient='h')
    plt.title("Linear Regression Coefficients")
    plt.xlabel("Coefficient Value (Standardized)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()


def random_forest_modeling(train_df, test_df):
    """
    Trains a Random Forest model with hyperparameter tuning using GridSearchCV on train_df,
    then evaluates on test_df.
    Non-numeric columns (e.g., 'extruder_location', 'protein_source') are dropped.
    """
    # Drop non-numeric columns from features
    X_train = train_df.drop(['firmness', 'extruder_location', 'protein_source'], axis=1)
    y_train = train_df['firmness']
    X_test = test_df.drop(['firmness', 'extruder_location', 'protein_source'], axis=1)
    y_test = test_df['firmness']

    # Define parameter grid for tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, scoring='r2')
    grid_search.fit(X_train, y_train)

    best_rf = grid_search.best_estimator_
    print("Best hyperparameters:", grid_search.best_params_)

    # Predict on the test set
    y_pred = best_rf.predict(X_test)

    # Evaluation metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Tuned Random Forest Test RMSE: {rmse:.4f}")
    print(f"Tuned Random Forest Test MAE: {mae:.4f}")
    print(f"Tuned Random Forest Test R²: {r2:.4f}")

    # Plot Prediction vs Actual
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', label='Ideal')
    plt.xlabel('Actual Firmness')
    plt.ylabel('Predicted Firmness')
    plt.title("Random Forest (Tuned): Predicted vs Actual Firmness")
    plt.legend()
    plt.show()

    # Residual Plot
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel("Residuals")
    plt.title("Residual Distribution (Tuned Random Forest)")
    plt.show()

    # Feature Importance Plot
    importances = best_rf.feature_importances_
    feature_names = X_train.columns
    sorted_idx = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[sorted_idx], y=feature_names[sorted_idx])
    plt.title("Feature Importance (Tuned Random Forest)")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()
