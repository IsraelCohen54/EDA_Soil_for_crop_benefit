import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from scipy import stats

DIV_BY_ZERO_SAFEGUARD = 1e-5


def encode_categorical_columns(df):
    for col in ['extruder_location', 'protein_source']:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col])
    return df


def normalize_by_city(data, normalization_method):
    """
    Normalize numeric columns in the DataFrame separately for each city.

    :param data: pd.DataFrame - Input DataFrame with numeric columns and 'extruder_location'.
    :param normalization_method: str - 'standard' for StandardScaler, 'minmax' for MinMaxScaler.
    :return: pd.DataFrame - Normalized DataFrame with preserved non-numeric data.
    """
    if normalization_method == 'standard':
        scaler_cls = StandardScaler
    elif normalization_method == 'minmax':
        scaler_cls = MinMaxScaler
    else:
        raise ValueError("method must be 'standard' or 'minmax'")

    city_col = 'extruder_location'
    encoded_categorical_cols = ['extruder_location_encoded', 'protein_source_encoded']

    if city_col not in data.columns:
        raise ValueError(f"Column '{city_col}' not found in DataFrame.")

    # Select numeric columns, excluding encoded categorical columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in encoded_categorical_cols]

    if not numeric_cols:
        raise ValueError("No valid numeric columns found for normalization.")

    # Collect normalized groups by city
    normalized_groups = []

    for city, group in data.groupby(city_col):
        scaler = scaler_cls()
        group_copy = group.copy()

        # Normalize only valid numeric columns
        group_copy[numeric_cols] = scaler.fit_transform(group_copy[numeric_cols])

        normalized_groups.append(group_copy)
        print(f"Normalized {len(group)} row(s) for city: {city}")

    # Concatenate all normalized groups once, reset index
    normalized_data = pd.concat(normalized_groups, axis=0).reset_index(drop=True)

    return normalized_data



def remove_duplicated_data(raw_data):
    initial_count = len(raw_data)
    num_duplicates = raw_data.duplicated(keep='first').sum()

    cleaned_dup_data = raw_data.drop_duplicates(keep='first')

    print(f"{num_duplicates} fully duplicated rows removed from Initial rows: {initial_count}")

    return cleaned_dup_data


def remove_rows_of_two_and_up_vacant_columns(data):
    # Count missing values per row
    mask = data.isnull().sum(axis=1) >= 2
    print(f"Removing {mask.sum()} row(s) with 2 or more missing values.")

    # Drop those rows
    cleaned_data = data[~mask]
    return cleaned_data


def remove_rows_without_city_data(data):
    mask = data['extruder_location'].isnull()
    print(f"Removing {mask.sum()} row(s) missing city data.")

    cleaned_data = data[~mask]
    return cleaned_data


def remove_dup_excluding_vacant_cell_column(raw_data):
    # Find rows with exactly 1 missing value
    rows_with_one_missing = raw_data[raw_data.isnull().sum(axis=1) == 1]

    print(f"Found {len(rows_with_one_missing)} row(s) with exactly 1 missing value.")

    indices_to_drop = []

    for idx, row in rows_with_one_missing.iterrows():
        # Create a mask of not-null columns
        not_null_cols = row.notnull()

        # Filter raw_data to rows that match on all not-null columns
        matching_rows = raw_data.loc[
            (raw_data[not_null_cols.index[not_null_cols]] == row[not_null_cols]).all(axis=1)
        ]

        # Remove the current row itself from the matching set
        matching_rows = matching_rows.drop(index=idx, errors='ignore')

        if not matching_rows.empty:
            # If there's at least one matching row (i.e., duplicate except for missing value)
            print(f"Row {idx} is a partial duplicate. Marked for removal.")
            indices_to_drop.append(idx)

    # Drop the identified rows
    cleaned_data = raw_data.drop(index=indices_to_drop)
    print(f"Removed {len(indices_to_drop)} partial duplicate row(s).")
    return cleaned_data


def avg_by_city_vacant_unique_data(raw_data):
    # Find rows with exactly 1 missing value
    rows_with_one_missing = raw_data[raw_data.isnull().sum(axis=1) == 1]

    print(f"Filling {len(rows_with_one_missing)} row(s) with city-based averages...")

    for idx, row in rows_with_one_missing.iterrows():
        # Find the column that is missing in this row
        missing_col = row[row.isnull()].index[0]  # Get name of the missing column

        # Get the city value for grouping
        city_value = row['extruder_location']

        # Filter data to same city & non-null for missing_col
        city_rows = raw_data[
            (raw_data['extruder_location'] == city_value) & (raw_data[missing_col].notnull())
            ]

        # Calculate city-based mean for the missing column
        city_mean = city_rows[missing_col].mean()

        if pd.isnull(city_mean):
            print(f"City {city_value} has no valid data for {missing_col}. Cannot fill row {idx}.")
            continue

        # Fill the missing value
        raw_data.at[idx, missing_col] = city_mean

        print(f"Filled row {idx}: {missing_col} with avg={city_mean:.2f} for city={city_value}")

    return raw_data


def fix_vacant_data(no_fully_dup_data):
    """
    logic is:
    A. if row has 2 and up vacant cell, too much, delete rows. (func - remove_rows_of_two_and_up_vacant_columns)
    B. if row has no city value - delete, too important. (func - remove_rows_without_city_data)
    C. if row has 1 vacant cell, and the row has identical row data except the vacant column
    - remove, as it won't add much (and would even make more weight for data already existed using avg method)
    (func - remove_dup_excluding_vacant_cell_column)
    D. if row include the 1 vacant cell is unique, avg would help to fill out the vacant data
       *** avg of the same city values *** to include its unique data in the results.
    (func - avg_vacant_unique_data)

    :param no_fully_dup_data: the data is cleaned for all fully duplicated rows (not including vacant cells)
    :return: cleaned data
    """
    cleaned_data_too_much_vacant_rows = remove_rows_of_two_and_up_vacant_columns(no_fully_dup_data)
    cleaned_data_no_city_inserted = remove_rows_without_city_data(cleaned_data_too_much_vacant_rows)

    data_no_dup = remove_dup_excluding_vacant_cell_column(cleaned_data_no_city_inserted)
    data = avg_by_city_vacant_unique_data(data_no_dup)
    return data


def detect_outliers_zscore(df, column, threshold=3):
    """
    Detects outliers in a given column using the z-score method.
    Returns a boolean series where True indicates an outlier.
    """
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    # Reindex to match the df index (fill missing rows with False)
    outlier_mask = pd.Series(False, index=df.index)
    outlier_mask.loc[df[column].dropna().index] = z_scores > threshold
    return outlier_mask


def remove_outliers(df, columns, threshold=3):
    """
    Remove rows that contain outliers in any of the specified columns.
    """
    mask = pd.Series(False, index=df.index)
    for col in columns:
        mask = mask | detect_outliers_zscore(df, col, threshold)
    df_clean = df[~mask].reset_index(drop=True)
    print(f"Removed {mask.sum()} rows as outliers in columns: {columns}")
    return df_clean


def clean_the_data(raw_data):
    raw_data.columns = raw_data.columns.str.strip()
    no_fully_dup_df = remove_duplicated_data(raw_data)
    no_dup_nor_vacant_df = fix_vacant_data(no_fully_dup_df)

    return no_dup_nor_vacant_df


def feature_engineering(df):
    df = df.copy()

    temp_cols = [
        'temperature_setpoint10cm',
        'temperature_setpoint20cm',
        'temperature_setpoint30cm',
        'temperature_setpoint40cm',
        'temperature_setpoint50cm',
        'temperature_setpoint60cm'
    ]

    # Feature 1: Average temperature
    df['avg_temp'] = df[temp_cols].mean(axis=1)

    # Feature 2: Temperature gradient (end - start)
    df['temp_gradient'] = df['temperature_setpoint60cm'] - df['temperature_setpoint10cm']

    # Feature 3: Temperature standard deviation
    df['temp_std'] = df[temp_cols].std(axis=1)

    # Feature 4: Water input relative to average temp
    df['water_temp_ratio'] = df['water_input'] / (df['avg_temp'] + DIV_BY_ZERO_SAFEGUARD)

    # Feature 5: Screw RPM interaction with avg temp
    df['rpm_temp_prod'] = df['screw_rpm'] * df['avg_temp']

    # Feature 6: Screw RPM interaction with water input
    df['rpm_water_prod'] = df['screw_rpm'] * df['water_input']

    # Feature 7: rpm interaction with avg temp and water
    df['rpm_water_temp_prod'] = df['water_input'] * df['avg_temp'] * df['screw_rpm']

    return df
