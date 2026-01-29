import pandas as pd
import numpy as np
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
from sklearn.linear_model import LinearRegression
import time

# 1. HELPER FUNCTION: Find Player ID by Name
def get_player_id(name):
    nba_players = players.get_players()
    player = [p for p in nba_players if p['full_name'].lower() == name.lower()]
    if not player:
        raise ValueError(f"Could not find player: {name}")
    return player[0]['id']

# 2. HELPER FUNCTION: Convert "MM:SS" or integer minutes to float
def clean_minutes(min_val):
    if isinstance(min_val, str) and ':' in min_val:
        parts = min_val.split(':')
        return float(parts[0]) + float(parts[1])/60
    return float(min_val)

# --- SETTINGS ---
PLAYER_NAME = "LeBron James"
PROJECTED_MINUTES = 35

try:
    print(f"Fetching data for {PLAYER_NAME}...")
    pid = get_player_id(PLAYER_NAME)

    # 3. FETCH DATA (2024-25 Season)
    gamelog = playergamelog.PlayerGameLog(player_id=pid, season='2024')
    df = gamelog.get_data_frames()[0]

    if df.empty:
        print("No game data found for this player in the current season.")
    else:
        # 4. DATA CLEANING
        # Use our helper function to fix the 'int object has no attribute split' error
        df['MIN_FLOAT'] = df['MIN'].apply(clean_minutes)
        
        # Take the last 15 games for a better sample size
        recent_games = df.head(15).copy()
        
        X = recent_games[['MIN_FLOAT']] # Features
        y = recent_games['PTS']        # Target

        # 5. TRAIN THE MODEL
        model = LinearRegression()
        model.fit(X, y)

        # 6. MAKE PREDICTION
        prediction = model.predict([[PROJECTED_MINUTES]])

        print("\n" + "="*30)
        print(f"RESULTS FOR {PLAYER_NAME.upper()}")
        print(f"Games analyzed: {len(recent_games)}")
        print(f"If he plays {PROJECTED_MINUTES} mins tonight...")
        print(f"PROJECTED POINTS: {prediction[0]:.1f}")
        print("="*30)

except Exception as e:
    print(f"An error occurred: {e}")