import pandas as pd
import numpy as np
import xgboost as xgb
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog

def get_player_id(name):
    nba_players = players.get_players()
    player = [p for p in nba_players if p['full_name'].lower() == name.lower()]
    return player[0]['id'] if player else None

def clean_minutes(min_val):
    if isinstance(min_val, str) and ':' in min_val:
        parts = min_val.split(':')
        return float(parts[0]) + float(parts[1])/60
    return float(min_val)

def run_prediction():
    print("\n--- PRO NBA Point Projector (XGBoost + Poisson) ---")
    target_player = input("Enter Player Full Name (or 'quit'): ")
    
    if target_player.lower() == 'quit':
        return False

    pid = get_player_id(target_player)
    if pid is None:
        print(f"Error: Could not find '{target_player}'.")
        return True

    try:
        # 1. Fetch Data
        gamelog = playergamelog.PlayerGameLog(player_id=pid, season='2024')
        df = gamelog.get_data_frames()[0]

        if df.empty:
            print("No data found for this season.")
            return True

        # 2. Prepare Data
        df['MIN_FLOAT'] = df['MIN'].apply(clean_minutes)
        # We'll use the last 20 games—XGBoost likes more data than Linear Regression
        recent = df.head(20).copy()
        
        X = recent[['MIN_FLOAT']]
        y = recent['PTS']
        
        # 3. THE HYBRID MODEL
        # objective='count:poisson' handles the "counting" nature of points
        model = xgb.XGBRegressor(
            objective='count:poisson',
            n_estimators=50,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        
        model.fit(X, y)

        # 4. Predict
        mins_input = input(f"Expected minutes for {target_player}? ")
        expected_mins = float(mins_input)
        
        prediction_df = pd.DataFrame([[expected_mins]], columns=['MIN_FLOAT'])
        prediction = model.predict(prediction_df)

        # Poisson ensures the result is positive, so no clipping is needed!
        print(f"\n>>> PRO PREDICTION: {target_player} projected {prediction[0]:.1f} points.")
        print("-" * 45)

    except Exception as e:
        print(f"Error: {e}")
    
    return True

running = True
while running:
    running = run_prediction()