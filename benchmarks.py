import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
import warnings

# Import user's custom functions
from vus import get_vus, generate_binned_df
from utils import clean_nba_data
from constants import TEAM_NAME_MAP

def fit_dynamic_benchmark(df, model_features):
    """
    Fits a dynamic benchmark model by training a separate logistic regression
    for each time bin.

    Args:
        df (pd.DataFrame): The preprocessed and merged game data DataFrame.
                           Must include 'time_bin', 'actual_result', and feature columns.
        model_features (list): A list of column names to use as features for the model.

    Returns:
        pd.DataFrame: The input DataFrame with a new 'home_WP' column containing
                      the model's predictions.
    """
    # Create a copy to avoid modifying the original DataFrame
    model_df = df.copy()
    model_df['home_WP'] = 0.5  # Initialize with a default value

    # Get the unique time bins to iterate over
    time_bins = sorted(model_df['time_bin'].unique())

    print(f"Fitting model with features: {model_features} across {len(time_bins)} time bins...")

    for bin_index in time_bins:
        # Get data for the current time bin
        bin_data = model_df[model_df['time_bin'] == bin_index]
        
        # Ensure there's enough data to train
        if len(bin_data) < 10 or len(bin_data['actual_result'].unique()) < 2:
            continue # Skip bins with insufficient data or only one outcome class

        X_train = bin_data[model_features]
        y_train = bin_data['actual_result']

        # Fit the logistic regression model for this specific bin
        log_reg = LogisticRegression(solver='liblinear', random_state=42)
        log_reg.fit(X_train, y_train)

        # Predict probabilities for this bin's data
        # We want the probability of the positive class (1)
        predicted_probs = log_reg.predict_proba(X_train)[:, 1]

        # Update the 'home_WP' for the rows corresponding to this bin
        model_df.loc[model_df['time_bin'] == bin_index, 'home_WP'] = predicted_probs
        
    return model_df

game_df = pd.read_csv('nba_2019.csv')
cleaned_df = clean_nba_data(game_df)

cols_to_relink = ['game_num']
for col in ['game_date', 'home_team', 'away_team']:
    if col in game_df.columns:
        cols_to_relink.append(col)

# Use only the unique columns to avoid errors
lookup_df = game_df[list(set(cols_to_relink))].drop_duplicates(subset=['game_num']).copy()

# Merge the necessary columns back into the cleaned dataframe
full_game_df = pd.merge(cleaned_df, lookup_df, on='game_num', how='left')
full_game_df['date'] = pd.to_datetime(full_game_df['game_date']).dt.strftime('%Y-%m-%d')

elo_df = pd.read_csv('nba_elo.csv')
elo_df = elo_df[elo_df['season'] == 2019]
elo_df['home_team'] = elo_df['team1'].map(TEAM_NAME_MAP)
elo_df['away_team'] = elo_df['team2'].map(TEAM_NAME_MAP)
elo_merge_cols = ['home_team', 'away_team']
elo_df['date'] = pd.to_datetime(elo_df['date']).dt.strftime('%Y-%m-%d')


temp_merged = pd.merge(
    full_game_df,
    elo_df[elo_merge_cols + ['date', 'elo_prob1']],
    on=elo_merge_cols,
    how='left',
    suffixes=('_game', '_elo')
)

temp_merged['date_game'] = pd.to_datetime(temp_merged['date_game'], errors='coerce').dt.tz_localize(None)
temp_merged['date_elo'] = pd.to_datetime(temp_merged['date_elo'], errors='coerce').dt.tz_localize(None)

temp_merged['date_diff'] = (temp_merged['date_elo'] - temp_merged['date_game']).dt.days.abs()

# Filter for matches where the date difference is 1 day or less
correct_matches = temp_merged[temp_merged['date_diff'] <= 1].copy()

# In case of duplicates, keep the one with the smallest date difference
correct_matches.sort_values(by=['game_num', 'date_diff'], inplace=True)
final_matches = correct_matches.drop_duplicates(subset=['game_num'], keep='first')

# Merge the final matched elo_prob1 back into the main dataframe
merged_df = pd.merge(
    full_game_df,
    final_matches[['game_num', 'elo_prob1']],
    on='game_num',
    how='left'
)

merged_df['elo_prob1'].fillna(0.5, inplace=True)



# Bin the data once to be used by dynamic models
num_bins = 100
merged_df['time_bin'], bin_map_check = pd.qcut(merged_df['normalized_time'], num_bins, labels=False, duplicates='drop', retbins=True)

merged_df['score_diff'] = merged_df['home'] - merged_df['away']


# --- 5. Calculate VUS for each Benchmark ---
print("\n--- Calculating VUS for Benchmark Models ---")

# --- Static Benchmark: Coin-Flip (CF) ---
print("\nProcessing: Coin-Flip (CF) Model")
cf_df = merged_df.copy()
cf_df['home_WP'] = 0.5
binned_cf_df, cf_bin_map = generate_binned_df(cf_df, num_bins=num_bins)
vus_cf = get_vus(binned_cf_df, cf_bin_map, num_bins=len(cf_bin_map))
print(f"Coin-Flip (CF) VUS: {vus_cf:.4f}")

# --- Static Benchmark: Historic Home Win (HomeWP) ---
print("\nProcessing: Historic Home Win (HomeWP) Model")
homewp_df = merged_df.copy()
homewp_df['home_WP'] = 0.593
binned_homewp_df, homewp_bin_map = generate_binned_df(homewp_df, num_bins=num_bins)
vus_homewp = get_vus(binned_homewp_df, homewp_bin_map, num_bins=len(homewp_bin_map))
print(f"Historic Home Win (HomeWP) VUS: {vus_homewp:.4f}")


print("\nProcessing: Pre-game Relative Strength (PgRS) Model")
pgrs_df = merged_df.copy()
pgrs_df['home_WP'] = pgrs_df['elo_prob1']
binned_pgrs_df, pgrs_bin_map = generate_binned_df(pgrs_df, num_bins=num_bins)
binned_pgrs_df.dropna(subset=['avg_home_WP'], inplace=True)
vus_pgrs = get_vus(binned_pgrs_df, pgrs_bin_map, num_bins=len(pgrs_bin_map))
print(f"Pre-game Relative Strength (PgRS) VUS: {vus_pgrs:.4f}")


# --- Dynamic Benchmark: Score Difference (ScD) ---
print("\nProcessing: Score Difference (ScD) Model")
scd_model_df = fit_dynamic_benchmark(merged_df, model_features=['score_diff'])
binned_scd_df, scd_bin_map = generate_binned_df(scd_model_df, num_bins=num_bins)
binned_scd_df.dropna(subset=['avg_home_WP'], inplace=True)
vus_scd = get_vus(binned_scd_df, scd_bin_map, num_bins=len(scd_bin_map))
print(f"Score Difference (ScD) VUS: {vus_scd:.4f}")

print("\nProcessing: Pregame RS + Score Difference (PgRSScD) Model")
pgrsscd_model_df = fit_dynamic_benchmark(merged_df, model_features=['elo_prob1', 'score_diff'])
binned_pgrsscd_df, pgrsscd_bin_map = generate_binned_df(pgrsscd_model_df, num_bins=num_bins)
binned_pgrsscd_df.dropna(subset=['avg_home_WP'], inplace=True)
vus_pgrsscd = get_vus(binned_pgrsscd_df, pgrsscd_bin_map, num_bins=len(pgrsscd_bin_map))
print(f"Pregame RS + Score Difference (PgRSScD) VUS: {vus_pgrsscd:.4f}")

