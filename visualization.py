import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot_correlation_heatmap(correlations):
    # Create the heatmap
    plt.figure(figsize=(12, 8))  # Set figure size
    sns.heatmap(correlations, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1)

    # Adjust the plot to move labels to the right and up
    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.28)

    plt.title("Correlation Heatmap")
    plt.show()


"""
def show_best_temperature_setpoint_fitness_correlated(df, city_labels):
    sns.scatterplot(x='temp_sp40cm', y='firmness', hue=city_labels, data=df)
    plt.title('40cm vs Firmness by City')
    plt.show()

    sns.histplot(data=df, x='firmness', hue=city_labels, kde=True)
    plt.title("Firmness Distribution by City")
    plt.show()

    for city_code, city_name in zip(df['extruder_location_encoded'].unique(), city_labels.unique()):
        city_data = df[df['extruder_location_encoded'] == city_code]
        corr = city_data['temp_sp40cm'].corr(city_data['firmness'])
        print(f"Correlation between 40cm and firmness in {city_name}: {corr:.3f}")"""


def feature_selection(df, target_column, threshold, plot_heatmap):
    """
    Select features based on their Pearson correlation with the target column.

    :param df: pd.DataFrame - The input DataFrame containing features and the target column.
    :param target_column: str - The target column name for correlation.
    :param threshold: float - Minimum absolute correlation value to keep a feature.
    :param plot_heatmap: bool - Whether to plot the correlation heatmap for selected features.
    :return: pd.DataFrame - DataFrame containing only the selected features and target column.
    """
    # Ensure target column exists in DataFrame
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the DataFrame.")

    # select numeric columns (include target column and categorical encoded)
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Drop categorical encodings columns
    categorical_encoded_columns = ['extruder_location_encoded', 'protein_source_encoded']
    numeric_columns = [col for col in numeric_columns if col not in categorical_encoded_columns]

    # Remove constant columns (zero variance) #  todo check
    df_cleaned = df[numeric_columns].loc[:, df[numeric_columns].nunique() > 1]

    # Handle case where no numeric columns are left after removing constant ones
    if df_cleaned.empty:
        raise ValueError("No numeric columns left after removing constant columns.")

    # Calculate Pearson correlation to target
    correlations = df_cleaned.corrwith(df[target_column]).abs()

    selected_features = correlations[correlations > threshold].index.tolist()

    if not selected_features:
        raise ValueError(f"No features found with correlation above {threshold} with target column '{target_column}'.")

    if plot_heatmap:
        # Uncomment to see the whole data correlation
        # plot_correlation_heatmap(df_cleaned.corr())
        plot_correlation_heatmap(df_cleaned[selected_features].corr())

    # Reinsert categorical data
    selected_features.extend(categorical_encoded_columns)

    return df[selected_features]


def plot_counts(df):
    # Count plots for extruder location and protein source
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.countplot(x='extruder_location', data=df, ax=ax[0])
    ax[0].set_title("Counts by Extruder Location")
    sns.countplot(x='protein_source', data=df, ax=ax[1])
    ax[1].set_title("Counts by Protein Source")
    plt.tight_layout()
    plt.show()


def boxplot_firmness_by_category(df):
    # Boxplots for firmness by city and by protein source
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(x='extruder_location', y='firmness', data=df, ax=ax[0])
    ax[0].set_title("Firmness by Extruder Location")
    sns.boxplot(x='protein_source', y='firmness', data=df, ax=ax[1])
    ax[1].set_title("Firmness by Protein Source")
    plt.tight_layout()
    plt.show()


def lineplot_avg_temp_along_barrel(df, group_by='extruder_location'):
    """
    Plots the average temperature at each sensor position along the barrel.
    `group_by` can be set to 'extruder_location' or 'protein_source'.
    """
    # Melt the temperature columns
    temp_cols = [col for col in df.columns if "temp_sp" in col]
    df_melt = df.melt(id_vars=[group_by], value_vars=temp_cols,
                      var_name='sensor_position', value_name='temperature')

    # Clean sensor_position to extract numeric position (e.g., 10cm, 20cm, etc.)
    df_melt['position'] = df_melt['sensor_position'].str.extract('(\d+)').astype(int)

    # Plot: using mean temperature per sensor position per group
    df_group = df_melt.groupby([group_by, 'position'])['temperature'].mean().reset_index()

    plt.figure(figsize=(10, 6))
    sns.lineplot(x='position', y='temperature', hue=group_by, data=df_group, marker='o')
    plt.xlabel("Sensor Position (cm)")
    plt.ylabel("Average Temperature")
    plt.title("Average Temperature Along Barrel by " + group_by.title())
    plt.show()


def box_violin_temp_by_category(df, category='extruder_location'):
    """
    Plots boxplots and violin plots for each temperature sensor by the given category.
    """
    temp_cols = [col for col in df.columns if "temp_sp" in col]
    df_melt = df.melt(id_vars=[category], value_vars=temp_cols,
                      var_name='sensor_position', value_name='temperature')

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(x='sensor_position', y='temperature', hue=category, data=df_melt)
    plt.title("Boxplot of Temperature Setpoints by " + category.title())
    plt.subplot(1, 2, 2)
    sns.violinplot(x='sensor_position', y='temperature', hue=category, data=df_melt, split=True)
    plt.title("Violin Plot of Temperature Setpoints by " + category.title())
    plt.tight_layout()
    plt.show()


def plot_feature_distributions(df, features, hue_column=None):
    """
    Plot histograms with KDE for a list of features.
    Optionally split by a hue column (e.g., protein_source or extruder_location).
    """
    n_features = len(features)
    plt.figure(figsize=(5 * n_features, 4))
    for i, feature in enumerate(features):
        plt.subplot(1, n_features, i+1)
        if hue_column:
            sns.histplot(data=df, x=feature, hue=hue_column, kde=True, element='step', stat='density')
        else:
            sns.histplot(data=df, x=feature, kde=True)
        plt.title(feature)
    # plt.tight_layout()
    plt.show()


def protein_firmness_city_rpm_facet_viz(df, city_labels, protein_labels):
    df_copy = df.copy()
    df_copy['City'] = city_labels
    df_copy['Protein'] = protein_labels

    g = sns.FacetGrid(df_copy, col='Protein', hue='City', palette='Set2', height=5, aspect=1.2)
    g.map_dataframe(sns.scatterplot, x='rpm_water_temp_prod', y='firmness')
    g.add_legend(title='City')
    g.set_axis_labels("Screw RPM", "Firmness")
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle('Firmness vs rpm_water_temp by Protein Source and City')
    plt.show()


def pairwise_firmness_screw_avgTemp_temp40(df, city_labels):  # rpm_column_type
    key_vars = ['firmness', 'rpm_water_temp_prod', 'avg_temp', 'temp_sp40cm']
    df_copy = df[key_vars].copy()
    df_copy['City'] = city_labels  # Add city context for cluster detection

    sns.pairplot(df_copy, hue='City', palette='Set2', diag_kind='kde', corner=True)
    plt.suptitle("Pairplot: Firmness, rpm_water_temp, Avg Temp, Temp 40cm by City", y=1.03)
    plt.tight_layout()
    plt.show()


def plot_firmness_vs_all_temps_by_city(df, city_labels):
    """
    Creates scatterplots of firmness vs. each temperature setpoint (columns starting with 'temp_sp')
    grouped by city.

    Parameters:
        df (DataFrame): DataFrame containing numeric features.
        city_labels (Series): Series of city names (e.g., mapped from 'extruder_location_encoded').
    """
    # Get all temperature sensor columns (assuming they start with 'temp_sp')
    temp_cols = [col for col in df.columns if col.startswith('temp_sp')]
    num_plots = len(temp_cols)

    # Create subplots
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5), sharey=True)

    # If only one temperature column exists, axes may not be a list
    if num_plots == 1:
        axes = [axes]

    for i, col in enumerate(temp_cols):
        sns.scatterplot(x=col, y='firmness', hue=city_labels, data=df, ax=axes[i])
        axes[i].set_title(f'{col} vs Firmness')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Firmness")

    plt.tight_layout()
    plt.show()


def pairwise_temperatures_40_50_60cm_with_firmness(df, protein_labels):
    """
    firmness and these temperature setpoints do not differ significantly between the protein sources:
    """
    key_vars = ['firmness', 'temp_sp40cm', 'temp_sp50cm', 'temp_sp60cm']
    df_copy = df[key_vars].copy()
    df_copy['Protein'] = protein_labels

    sns.pairplot(df_copy, hue='Protein', palette='Set1', diag_kind='kde', corner=True)
    plt.suptitle("Pairplot: Firmness and Temp Setpoints (40/50/60cm) by Protein Source", y=1.03)
    plt.tight_layout()
    plt.show()
