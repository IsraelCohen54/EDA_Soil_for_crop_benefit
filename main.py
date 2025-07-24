import pandas as pd
from sklearn.model_selection import train_test_split

import ml_models
import preprocessing
import visualization
from sensor_removal_evaluation import evaluate_sensor_removal_norm

FILE_PATH = "raw_data.csv"

raw_data = pd.read_csv(FILE_PATH)

# ~~ Preprocessing: ~~
raw_data_no_dup_or_whitespace = preprocessing.clean_the_data(raw_data)
raw_data_encoded_categories = preprocessing.encode_categorical_columns(raw_data_no_dup_or_whitespace)
df_with_features = preprocessing.feature_engineering(raw_data_encoded_categories)
df_with_features.columns = [col.replace('temperature', 'temp') for col in df_with_features.columns]
df_with_features.columns = [col.replace('setpoint', 'sp') for col in df_with_features.columns]

# Remove outliers for selected numeric columns
numeric_cols_for_outlier = ['firmness', 'screw_rpm', 'temp_sp10cm', 'temp_sp20cm',
                            'temp_sp30cm', 'temp_sp40cm', 'temp_sp50cm', 'temp_sp60cm']
df_no_outliers = preprocessing.remove_outliers(df_with_features, numeric_cols_for_outlier, threshold=3)

# ( Split into Train/Test sets prior to normalization only for modeling )
train_df, test_df = train_test_split(df_no_outliers, test_size=0.3, random_state=42)

# Normalization options (apply for visualization only):
# 1. 0-1 scaling:
# df_normalized = preprocessing.normalize_by_city(df_no_outliers, normalization_method='minmax')

# 2. Z-score standardization, uncomment: {
df_normalized = preprocessing.normalize_by_city(df_no_outliers, normalization_method='standard')

# ~~ Visualisation ~~:
# df = df_normalized
# optional:
df = visualization.feature_selection(df_normalized, target_column='firmness', threshold=0.1, plot_heatmap=True)

# 1. Check counts to see if data are balanced
visualization.plot_counts(df_normalized)
# 2. Boxplots for firmness by extruder location and protein source
visualization.boxplot_firmness_by_category(df_normalized)
# 3. Line plot for average temperature along the barrel, grouped by city
visualization.lineplot_avg_temp_along_barrel(df_normalized, group_by='extruder_location')
# 4. Boxplots/Violin plots for temperature setpoints by protein source
visualization.box_violin_temp_by_category(df_normalized, category='protein_source')
# 5. Histograms/KDE plots for new features
new_features = ['temp_std', 'temp_gradient', 'rpm_temp_prod', 'rpm_water_prod', 'rpm_water_temp_prod']
visualization.plot_feature_distributions(df_normalized, new_features, hue_column='protein_source')


city_labels = df['extruder_location_encoded'].map({0: 'Haifa', 1: 'TLV'})
visualization.plot_firmness_vs_all_temps_by_city(df, city_labels)
# visualisation.show_best_temperature_setpoint_fitness_correlated(df, city_labels)


# Some more visualization, I think they dont give much more than show how balanced is the data.
# protein_labels = df['protein_source_encoded'].map({0: 'Pea', 1: 'Soy'})
# #  show protein balanced data by city:
# visualization.protein_firmness_city_rpm_facet_viz(df, city_labels, protein_labels)
# optional to check some more balance data
# visualization.pairwise_firmness_screw_avgTemp_temp40(df, city_labels)
# visualization.pairwise_temperatures_40_50_60cm_with_firmness(df, protein_labels)

# ~~ ML models: ~~
ml_models.linear_regression_modeling(train_df, test_df)
ml_models.random_forest_modeling(train_df, test_df)

# ~~ Sensor removal evaluation: ~~
# Define temperature sensor columns (using the column names after renaming has been done)
sensor_columns = ['temp_sp10cm', 'temp_sp20cm', 'temp_sp30cm',
                  'temp_sp40cm', 'temp_sp50cm', 'temp_sp60cm']

# Evaluate the impact of removing each sensor using the pre-split train and test sets.
removal_results, best_sensor_to_remove \
    = evaluate_sensor_removal_norm(train_df, test_df, sensor_columns, target='firmness')


# ~~ general statistics: ~~
# df_no_outliers chosen for df prior to normalization:
with pd.option_context('display.max_columns', None, 'display.width', 1000):
    stats_summary = df_no_outliers.describe()
    stats_city = df_no_outliers.groupby('extruder_location')['firmness'].describe()
    stats_protein = df_no_outliers.groupby('protein_source')['firmness'].describe()

    with open("EDA_statistics.csv", "w", encoding="utf-8") as f:
        f.write("Summary Statistics for Key Variables:\n")
        stats_summary.to_csv(f)
        f.write("\nFirmness Stats by City:\n")
        stats_city.to_csv(f)
        f.write("\nFirmness Stats by Protein Source:\n")
        stats_protein.to_csv(f)
