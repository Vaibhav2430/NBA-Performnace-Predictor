import pandas as pd
import numpy as np
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
from sklearn.linear_model import LinearRegression

# 1. HELPER: Find Player ID
def get_player_id(name):
    nba_players = players.get_players()
    player = [p for p in nba_players if p['full_name'].lower() == name.lower()]
    return player[0]['id'] if player else None

# 2. HELPER: Convert "MM:SS" strings or integers to float
def clean_minutes(min_val):
    if isinstance(min_val, str) and ':' in min_val:
        parts = min_val.split(':')
        return float(parts[0]) + float(parts[1])/60
    return float(min_val)

def run_prediction():
    print("\n--- NBA Player Point Projector ---")
    target_player = input("Enter Player Full Name (or 'quit' to stop): ")
    
    if target_player.lower() == 'quit':
        return False

    pid = get_player_id(target_player)
    if pid is None:
        print(f"Error: Could not find '{target_player}'. Check your spelling!")
        return True

    try:
        # 3. Fetch Data for 2024-25 Season
        print(f"Analyzing {target_player}'s recent performance...")
        gamelog = playergamelog.PlayerGameLog(player_id=pid, season='2024')
        df = gamelog.get_data_frames()[0]

        if df.empty:
            print("No data found for this player in the current season.")
            return True

        # 4. Data Cleaning
        df['MIN_FLOAT'] = df['MIN'].apply(clean_minutes)
        recent = df.head(15).copy() # Use last 15 games
        
        # 5. Train the Model
        # X must be a DataFrame with a name to match prediction later
        X = recent[['MIN_FLOAT']] 
        y = recent['PTS']
        
        model = LinearRegression()
        model.fit(X, y)

        # 6. User Input for Prediction
        mins_input = input(f"How many minutes do you expect {target_player} to play tonight? ")
        expected_mins = float(mins_input)

        # 7. Make Prediction (using a DataFrame to avoid UserWarnings)
        prediction_df = pd.DataFrame([[expected_mins]], columns=['MIN_FLOAT'])
        prediction_raw = model.predict(prediction_df)

        # 8. Prevent Negative Points
        # np.clip ensures the number never drops below 0.0
        final_points = np.clip(prediction_raw[0], 0, None)

        print(f"\n>>> RESULT FOR {target_player.upper()}")
        print(f"Based on 15 games, if playing {expected_mins} mins:")
        print(f"PROJECTED POINTS: {final_points:.1f}")
        print("-" * 40)

    except Exception as e:
        print(f"Something went wrong: {e}")
    
    return True

# Main Loop
running = True
while running:
    running = run_prediction()