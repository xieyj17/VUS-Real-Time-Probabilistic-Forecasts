import pandas as pd
import re

QUARTER_IN_SEC = 12*60

def time_to_seconds(time_str):

    # Handle cases where 'clock' might be an integer (seconds)
    if isinstance(time_str, (int, float)):
        return float(time_str)

    # Handle mm:ss format
    match_mm_ss = re.match(r'(\d+):(\d+)', str(time_str))
    if match_mm_ss:
        minutes = int(match_mm_ss.group(1))
        seconds = int(match_mm_ss.group(2))
        return minutes * 60 + seconds

    return float(time_str) # If it's a string like '615' and convertible to int


def time_left_in_quarter(second):
    return QUARTER_IN_SEC - second

def normalize_time(df):
    # Calculate the seconds passed within the current quarter for each row
    df['current_quarter_seconds_passed'] = df['clock'].apply(
        lambda x: time_left_in_quarter(time_to_seconds(x))
    )

    # Calculate the total seconds passed from the beginning of the game.
    # This sums seconds from all fully completed previous quarters
    # and the seconds passed in the current quarter.
    df['second_passed'] = (df['Quarter'] - 1) * QUARTER_IN_SEC + df['current_quarter_seconds_passed']
    
    df['normalized_time'] = df['second_passed'] / (4 * QUARTER_IN_SEC)
    # Remove the intermediate columns if they're not needed in the final output
    # df = df.drop(columns=[['current_quarter_seconds_passed', 'second_passed']])

    return df

def actual_results(df):
    # Ensure 'play_id' is numeric for max() operation, coercing errors to NaN
    df['play_id'] = pd.to_numeric(df['play_id'], errors='coerce')
    # Drop rows where play_id is NaN after conversion, as they cannot be used to find max
    df.dropna(subset=['play_id'], inplace=True)


    # --- Calculate Actual Result (Home Win/Loss) ---
    # Group by game_num and find the row with the maximum play_id for each game.
    # Then, use the 'home' and 'away' scores from that specific row to determine the winner.

    # Find the index of the row with the maximum 'play_id' for each game_num
    idx = df.groupby('game_num')['play_id'].idxmax()

    # Select these specific rows
    final_game_states = df.loc[idx]

    # Calculate 'actual_result' based on 'home' and 'away' scores in these final states
    # Assuming 'home' and 'away' columns represent the scores
    final_game_states['actual_result'] = final_game_states.apply(
        lambda row: 1 if row['home'] > row['away'] else 0, axis=1
    )

    # Create a mapping from game_num to actual_result
    actual_result_map = final_game_states.set_index('game_num')['actual_result'].to_dict()

    # Map the actual_result back to all rows in the original (filtered) DataFrame
    df['actual_result'] = df['game_num'].map(actual_result_map)

    # Sort data for proper time series, now by normalized_game_time
    df = df.sort_values(by=['game_num', 'normalized_time'], ascending=[True, True])

    return df

def clean_nba_data(input_df):
    df = normalize_time(input_df)
    df = actual_results(df)
    df = df[df["Quarter"] <= 4] # remove overtime
    df = df[["game_num", "normalized_time", "home_WP", "actual_result", "home", "away"]].drop_duplicates()
    return df
