import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
from typing import Tuple, List, Dict
import joblib
import warnings
warnings.filterwarnings('ignore')
# Import Player from models.py
from models import Player, Position

class PlayerPerformancePredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = [
            'price', 'total_points', 'points_per_game', 'form', 'minutes',
            'goals_scored', 'assists', 'clean_sheets', 'goals_conceded',
            'bonus', 'bps', 'influence', 'creativity', 'threat', 'ict_index',
            'expected_goals', 'expected_assists', 'selected_by_percent'
        ]

    def prepare_features(self, players: List[Player]) -> pd.DataFrame:
        """Convert player data to ML features"""
        data = []
        for player in players:
            row = {
                'id': player.id,
                'name': player.name,
                'position': player.position.value,
                'team': player.team,
                'price': player.price,
                'total_points': player.total_points,
                'points_per_game': player.points_per_game,
                'form': player.form,
                'minutes': player.minutes,
                'goals_scored': player.goals_scored,
                'assists': player.assists,
                'clean_sheets': player.clean_sheets,
                'goals_conceded': player.goals_conceded,
                'bonus': player.bonus,
                'bps': player.bps,
                'influence': player.influence,
                'creativity': player.creativity,
                'threat': player.threat,
                'ict_index': player.ict_index,
                'expected_goals': player.expected_goals,
                'expected_assists': player.expected_assists,
                'selected_by_percent': player.selected_by_percent
            }
            data.append(row)
        return pd.DataFrame(data)

    def train_position_models(self, df: pd.DataFrame, target_column: str = 'points_per_game'):
        """Train separate models for each position"""
        positions = df['position'].unique()
        for position in positions:
            pos_data = df[df['position'] == position].copy()
            if len(pos_data) < 10:  # Skip if insufficient data
                continue
            X = pos_data[self.feature_columns]
            y = pos_data[target_column]
            # Handle missing values
            X = X.fillna(X.median())
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            # Train ensemble of models
            models = {
                'rf': RandomForestRegressor(n_estimators=100, random_state=42),
                'gbr': GradientBoostingRegressor(random_state=42),
                'xgb': xgb.XGBRegressor(random_state=42)
            }
            position_models = {}
            for name, model in models.items():
                model.fit(X_scaled, y)
                position_models[name] = model
            self.models[position] = position_models
            self.scalers[position] = scaler

    def predict_next_gameweek_points(self, players: List[Player]) -> Dict[int, float]:
        """Predict next gameweek points for all players"""
        df = self.prepare_features(players)
        predictions = {}
        for position in df['position'].unique():
            if position not in self.models:
                continue
            pos_players = df[df['position'] == position].copy()
            X = pos_players[self.feature_columns].fillna(pos_players[self.feature_columns].median())
            X_scaled = self.scalers[position].transform(X)
            # Ensemble prediction
            pos_predictions = []
            for model in self.models[position].values():
                pred = model.predict(X_scaled)
                pos_predictions.append(pred)
            # Average ensemble predictions
            ensemble_pred = np.mean(pos_predictions, axis=0)
            for idx, player_id in enumerate(pos_players['id']):
                predictions[player_id] = max(0, ensemble_pred[idx])  # Ensure non-negative
        return predictions