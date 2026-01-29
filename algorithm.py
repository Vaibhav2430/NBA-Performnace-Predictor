import pandas as pd
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
from sklearn.linear_model import LinearRegression
import numpy as np
import time

# 1. FIND PLAYER ID
def get_player_id(name):
    nba_players = players.get_players()
    player = [p for p in nba_players if p['full_name'].lower() == name.lower()][0]
    return player['id']

# 2. GET RECENT GAME LOGS
player_name = "LeBron James"
pid = get_player_id(player_name)

# Fetch games from the current 2024-25 season
gamelog = playergamelog.PlayerGameLog(player_id=pid, season='2024')
df = gamelog.get_data_frames()[0]

# 3. PREPARE DATA FOR MODEL
# We'll use 'MIN' (Minutes) to predict 'PTS' (Points)
# We take the last 10 games to train our basic model
recent_games = df.head(10).copy()

# Convert Minutes string "32:15" to a float 32.25
recent_games['MIN_FLOAT'] = recent_games['MIN'].apply(lambda x: float(x.split(':')[0]) + float(x.split(':')[1])/60)

X = recent_games[['MIN_FLOAT']]
y = recent_games['PTS']

# 4. TRAIN & PREDICT
model = LinearRegression()
model.fit(X, y)

# Predict for next game assuming 35 minutes of play
prediction = model.predict([[35]])

print(f"--- {player_name} Prediction ---")
print(f"Based on his last 10 games, if he plays 35 mins, he is projected to score: {prediction[0]:.1f} pts")