from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


def evaluate_sensor_removal_norm(train_df, test_df, sensor_cols, target='firmness'):
    """
    Evaluates the impact of removing each sensor (from sensor_cols) on model performance,
    using a normalized Linear Regression model.

    For each sensor, the function:
      - Removes that sensor from the feature set.
      - Applies StandardScaler to the remaining sensors.
      - Trains a Linear Regression model and computes R² on the test set.
      - Calculates ΔR² (the drop in R² compared to using all sensors).

    It then plots:
      - The R² values per removed sensor.
      - The performance drop (ΔR²) per removed sensor.

    Returns:
        results (dict): Mapping of each sensor to (R² without sensor, ΔR²).
        best_sensor (str): The sensor whose removal causes the smallest drop in performance.
    """
    results = {}

    # Full model performance with all sensors
    scaler_full = StandardScaler()
    X_train_full = scaler_full.fit_transform(train_df[sensor_cols])
    X_test_full = scaler_full.transform(test_df[sensor_cols])

    model_full = LinearRegression()
    model_full.fit(X_train_full, train_df[target])
    y_pred_full = model_full.predict(X_test_full)
    full_r2 = r2_score(test_df[target], y_pred_full)
    print(f"Full model R² using all sensors: {full_r2:.4f}")

    # Evaluate each sensor removal
    for sensor in sensor_cols:
        remaining_sensors = [s for s in sensor_cols if s != sensor]
        scaler = StandardScaler()
        X_train_sub = scaler.fit_transform(train_df[remaining_sensors])
        X_test_sub = scaler.transform(test_df[remaining_sensors])

        model = LinearRegression()
        model.fit(X_train_sub, train_df[target])
        y_pred_sub = model.predict(X_test_sub)
        r2_val = r2_score(test_df[target], y_pred_sub)
        delta = full_r2 - r2_val
        results[sensor] = (r2_val, delta)
        print(f"Removing {sensor}: R² = {r2_val:.4f}, ΔR² = {delta:.4f}")

    # Plot R² for each removed sensor
    sensors = list(results.keys())
    r2_values = [results[s][0] for s in sensors]
    delta_values = [results[s][1] for s in sensors]

    plt.figure(figsize=(8, 6))
    plt.bar(sensors, r2_values, color='skyblue')
    plt.xlabel("Removed Sensor")
    plt.ylabel("R² on Test Set")
    plt.title("R² Performance per Removed Sensor")
    plt.show()

    # Plot the drop in performance (ΔR²)
    plt.figure(figsize=(8, 6))
    plt.bar(sensors, delta_values, color='salmon')
    plt.xlabel("Removed Sensor")
    plt.ylabel("ΔR² (Full R² - Removed R²)")
    plt.title("Performance Drop (ΔR²) per Removed Sensor")
    plt.show()

    # The sensor with the smallest drop (ΔR²) is the best candidate for removal.
    best_sensor = min(results, key=lambda s: results[s][1])
    print("Best sensor candidate for removal:", best_sensor)

    return results, best_sensor
