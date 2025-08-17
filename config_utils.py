import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import streamlit as st

# config.py - Configuration settings
class FPLConfig:
    """Configuration settings for FPL AI Assistant"""
    
    # API Settings
    FPL_BASE_URL = "https://fantasy.premierleague.com/api/"
    API_TIMEOUT = 30
    CACHE_DURATION = 3600  # 1 hour in seconds
    
    # Team Constraints
    TEAM_CONSTRAINTS = {
        'total_players': 15,
        'starting_players': 11,
        'max_players_per_team': 3,
        'position_limits': {
            'Goalkeeper': 2,
            'Defender': 5,
            'Midfielder': 5,
            'Forward': 3
        },
        'formation_limits': {
            'min_defenders': 3,
            'max_defenders': 5,
            'min_midfielders': 2,
            'max_midfielders': 5,
            'min_forwards': 1,
            'max_forwards': 3
        }
    }
    
    # Budget Settings
    DEFAULT_BUDGET = 100.0  # Million pounds
    MIN_PLAYER_COST = 3.9
    MAX_PLAYER_COST = 15.0
    
    # ML Model Settings
    ML_SETTINGS = {
        'ensemble_weights': {
            'random_forest': 0.4,
            'gradient_boost': 0.4,
            'linear_regression': 0.2
        },
        'feature_weights': {
            'total_points': 0.3,
            'form_float': 0.25,
            'points_per_game': 0.2,
            'minutes': 0.1,
            'influence': 0.05,
            'creativity': 0.05,
            'threat': 0.05
        },
        'position_multipliers': {
            'Goalkeeper': 0.8,
            'Defender': 0.9,
            'Midfielder': 1.1,
            'Forward': 1.2
        },
        'test_size': 0.2,
        'random_state': 42,
        'cross_val_folds': 5
    }
    
    # Genetic Algorithm Settings
    GA_SETTINGS = {
        'population_size': 100,
        'generations': 50,
        'crossover_rate': 0.8,
        'mutation_rate': 0.1,
        'tournament_size': 3,
        'elite_size': 10
    }
    
    # RAG System Settings
    RAG_SETTINGS = {
        'embedding_model': 'gemini-embedding-001',
        'vector_db_path': './fpl_knowledge_base',
        'collection_name': 'fpl_knowledge',
        'max_context_length': 2000,
        'similarity_threshold': 0.7,
        'top_k_results': 3,
        'euriai_model': 'gemini-2.5-pro',  # Default EuriAI model
        'use_euriai_embeddings': True
    }
    
    # Chip Strategy Settings
    CHIP_STRATEGY = {
        'wildcard_1': {'min_gw': 7, 'max_gw': 10, 'priority': 'high'},
        'wildcard_2': {'min_gw': 19, 'max_gw': 25, 'priority': 'high'},
        'free_hit': {'target_gws': [18, 29], 'priority': 'medium'},
        'bench_boost': {'target_gws': [26, 34, 37], 'priority': 'high'},
        'triple_captain': {'target_gws': [26, 35], 'priority': 'high'}
    }
    
    # UI Settings
    UI_SETTINGS = {
        'page_title': 'FPL AI Assistant',
        'page_icon': '⚽',
        'layout': 'wide',
        'theme_color': '#37003c',
        'max_file_size': 200,
        'chart_height': 400
    }
    
    @classmethod
    def load_custom_config(cls, config_path: str = 'custom_config.json'):
        """Load custom configuration from JSON file"""
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    custom_config = json.load(f)
                
                # Update class attributes with custom values
                for section, values in custom_config.items():
                    if hasattr(cls, section):
                        current_attr = getattr(cls, section)
                        if isinstance(current_attr, dict):
                            current_attr.update(values)
                        else:
                            setattr(cls, section, values)
                
                logging.info(f"Loaded custom configuration from {config_path}")
            except Exception as e:
                logging.error(f"Error loading custom config: {e}")
    
    @classmethod
    def save_config(cls, config_path: str = 'current_config.json'):
        """Save current configuration to JSON file"""
        config_dict = {}
        for attr in dir(cls):
            if not attr.startswith('_') and not callable(getattr(cls, attr)):
                config_dict[attr] = getattr(cls, attr)
        
        try:
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=4, default=str)
            logging.info(f"Configuration saved to {config_path}")
        except Exception as e:
            logging.error(f"Error saving config: {e}")

# utils.py - Utility functions
class DataValidator:
    """Validates and cleans FPL data"""
    
    @staticmethod
    def validate_player_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Validate and clean player data"""
        errors = []
        cleaned_df = df.copy()
        
        # Required columns
        required_cols = ['web_name', 'position', 'team_name', 'now_cost', 'total_points']
        missing_cols = [col for col in required_cols if col not in cleaned_df.columns]
        
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            return pd.DataFrame(), errors
        
        # Data type validation
        numeric_cols = ['now_cost', 'total_points', 'minutes', 'points_per_game']
        for col in numeric_cols:
            if col in cleaned_df.columns:
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
                null_count = cleaned_df[col].isnull().sum()
                if null_count > 0:
                    cleaned_df[col].fillna(0, inplace=True)
                    errors.append(f"Found {null_count} null values in {col}, filled with 0")
        
        # Cost validation
        if 'now_cost' in cleaned_df.columns:
            invalid_costs = cleaned_df[
                (cleaned_df['now_cost'] < FPLConfig.MIN_PLAYER_COST * 10) |
                (cleaned_df['now_cost'] > FPLConfig.MAX_PLAYER_COST * 10)
            ]
            if not invalid_costs.empty:
                errors.append(f"Found {len(invalid_costs)} players with invalid costs")
        
        # Position validation
        valid_positions = ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']
        if 'position' in cleaned_df.columns:
            invalid_positions = cleaned_df[~cleaned_df['position'].isin(valid_positions)]
            if not invalid_positions.empty:
                errors.append(f"Found {len(invalid_positions)} players with invalid positions")
                cleaned_df = cleaned_df[cleaned_df['position'].isin(valid_positions)]
        
        # Remove duplicates
        initial_count = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates(subset=['web_name'])
        duplicates_removed = initial_count - len(cleaned_df)
        if duplicates_removed > 0:
            errors.append(f"Removed {duplicates_removed} duplicate players")
        
        return cleaned_df, errors
    
    @staticmethod
    def validate_team_selection(team: List[Dict]) -> Tuple[bool, List[str]]:
        """Validate team selection against FPL rules"""
        errors = []
        
        if len(team) != FPLConfig.TEAM_CONSTRAINTS['total_players']:
            errors.append(f"Team must have {FPLConfig.TEAM_CONSTRAINTS['total_players']} players")
            return False, errors
        
        # Position constraints
        position_counts = {}
        team_counts = {}
        total_cost = 0
        
        for player in team:
            # Count positions
            pos = player.get('position', 'Unknown')
            position_counts[pos] = position_counts.get(pos, 0) + 1
            
            # Count teams
            team_name = player.get('team', 'Unknown')
            team_counts[team_name] = team_counts.get(team_name, 0) + 1
            
            # Sum costs
            total_cost += player.get('cost', 0)
        
        # Check position limits
        for pos, limit in FPLConfig.TEAM_CONSTRAINTS['position_limits'].items():
            count = position_counts.get(pos, 0)
            if count != limit:
                errors.append(f"Need exactly {limit} {pos}s, found {count}")
        
        # Check team limits
        max_per_team = FPLConfig.TEAM_CONSTRAINTS['max_players_per_team']
        for team_name, count in team_counts.items():
            if count > max_per_team:
                errors.append(f"Too many players from {team_name}: {count} (max {max_per_team})")
        
        # Check budget
        if total_cost > FPLConfig.DEFAULT_BUDGET:
            errors.append(f"Team cost £{total_cost:.1f}M exceeds budget £{FPLConfig.DEFAULT_BUDGET}M")
        
        return len(errors) == 0, errors

class PerformanceTracker:
    """Tracks application performance and usage metrics"""
    
    def __init__(self):
        self.metrics = {
            'api_calls': 0,
            'ml_predictions': 0,
            'optimizations_run': 0,
            'teams_generated': 0,
            'errors': []
        }
        self.start_time = datetime.now()
    
    def log_api_call(self, endpoint: str, response_time: float):
        """Log API call metrics"""
        self.metrics['api_calls'] += 1
        logging.info(f"API call to {endpoint} took {response_time:.2f}s")
    
    def log_ml_prediction(self, model_name: str, prediction_time: float, num_players: int):
        """Log ML prediction metrics"""
        self.metrics['ml_predictions'] += 1
        logging.info(f"{model_name} predicted {num_players} players in {prediction_time:.2f}s")
    
    def log_optimization(self, method: str, runtime: float, fitness_score: float):
        """Log optimization metrics"""
        self.metrics['optimizations_run'] += 1
        logging.info(f"{method} optimization completed in {runtime:.2f}s with fitness {fitness_score}")
    
    def log_error(self, error_type: str, error_message: str):
        """Log error occurrences"""
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'type': error_type,
            'message': error_message
        }
        self.metrics['errors'].append(error_info)
        logging.error(f"{error_type}: {error_message}")
    
    def get_summary(self) -> Dict:
        """Get performance summary"""
        runtime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            'session_runtime': runtime,
            'api_calls': self.metrics['api_calls'],
            'ml_predictions': self.metrics['ml_predictions'],
            'optimizations': self.metrics['optimizations_run'],
            'teams_generated': self.metrics['teams_generated'],
            'error_count': len(self.metrics['errors']),
            'avg_calls_per_minute': (self.metrics['api_calls'] / runtime) * 60 if runtime > 0 else 0
        }

class CacheManager:
    """Manages caching for FPL data and ML models"""
    
    def __init__(self, cache_dir: str = './cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def cache_key(self, prefix: str, *args) -> str:
        """Generate cache key from arguments"""
        key_parts = [str(arg) for arg in args]
        return f"{prefix}_{'_'.join(key_parts)}.json"
    
    def is_cache_valid(self, filepath: str, max_age_seconds: int = 3600) -> bool:
        """Check if cached file is still valid"""
        if not os.path.exists(filepath):
            return False
        
        file_age = time.time() - os.path.getmtime(filepath)
        return file_age < max_age_seconds
    
    def save_to_cache(self, data: Dict, cache_key: str):
        """Save data to cache"""
        filepath = os.path.join(self.cache_dir, cache_key)
        try:
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'data': data
            }
            with open(filepath, 'w') as f:
                json.dump(cache_data, f, indent=2, default=str)
            logging.info(f"Data cached to {filepath}")
        except Exception as e:
            logging.error(f"Error caching data: {e}")
    
    def load_from_cache(self, cache_key: str, max_age_seconds: int = 3600) -> Optional[Dict]:
        """Load data from cache if valid"""
        filepath = os.path.join(self.cache_dir, cache_key)
        
        if not self.is_cache_valid(filepath, max_age_seconds):
            return None
        
        try:
            with open(filepath, 'r') as f:
                cache_data = json.load(f)
            
            logging.info(f"Loaded data from cache: {cache_key}")
            return cache_data['data']
        except Exception as e:
            logging.error(f"Error loading cache: {e}")
            return None
    
    def clear_cache(self, pattern: Optional[str] = None):
        """Clear cache files matching pattern"""
        import glob
        
        if pattern:
            pattern_path = os.path.join(self.cache_dir, f"*{pattern}*")
            files_to_remove = glob.glob(pattern_path)
        else:
            files_to_remove = glob.glob(os.path.join(self.cache_dir, "*.json"))
        
        for filepath in files_to_remove:
            try:
                os.remove(filepath)
                logging.info(f"Removed cache file: {filepath}")
            except Exception as e:
                logging.error(f"Error removing cache file {filepath}: {e}")
        
        return len(files_to_remove)

class FixtureAnalyzer:
    """Analyzes fixture difficulty and scheduling"""
    
    def __init__(self):
        self.fixture_cache = {}
    
    def calculate_fixture_difficulty(self, team_fixtures: List[Dict]) -> Dict:
        """Calculate fixture difficulty scores"""
        difficulty_scores = {
            'next_5': 0,
            'next_10': 0,
            'season_remaining': 0,
            'home_away_balance': 0
        }
        
        home_count = 0
        total_difficulty = 0
        
        for i, fixture in enumerate(team_fixtures[:10]):  # Next 10 fixtures
            difficulty = fixture.get('difficulty', 3)  # Default to 3 (medium)
            is_home = fixture.get('is_home', True)
            
            # Weight recent fixtures more heavily
            weight = 1.0 if i >= 5 else 1.5
            total_difficulty += difficulty * weight
            
            if i < 5:
                difficulty_scores['next_5'] += difficulty
            
            if is_home:
                home_count += 1
        
        difficulty_scores['next_10'] = total_difficulty / 10
        difficulty_scores['home_away_balance'] = (home_count - 5) / 5  # -1 to 1 scale
        
        return difficulty_scores
    
    def identify_fixture_swings(self, all_team_fixtures: Dict) -> Dict:
        """Identify favorable and unfavorable fixture periods"""
        swing_periods = {
            'good_periods': [],
            'bad_periods': [],
            'dgw_candidates': [],
            'blank_gw_risks': []
        }
        
        for team, fixtures in all_team_fixtures.items():
            team_difficulty = self.calculate_fixture_difficulty(fixtures)
            
            if team_difficulty['next_5'] <= 2.0:
                swing_periods['good_periods'].append({
                    'team': team,
                    'period': 'next_5',
                    'avg_difficulty': team_difficulty['next_5']
                })
            elif team_difficulty['next_5'] >= 4.0:
                swing_periods['bad_periods'].append({
                    'team': team,
                    'period': 'next_5',
                    'avg_difficulty': team_difficulty['next_5']
                })
        
        return swing_periods

def setup_logging():
    """Set up application logging"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create logs directory
    os.makedirs('./logs', exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler('./logs/fpl_app.log'),
            logging.StreamHandler()
        ]
    )
    
    # Create specialized loggers
    loggers = {
        'data': logging.getLogger('fpl.data'),
        'ml': logging.getLogger('fpl.ml'),
        'optimization': logging.getLogger('fpl.optimization'),
        'api': logging.getLogger('fpl.api')
    }
    
    return loggers

def format_currency(amount: float, currency: str = "£") -> str:
    """Format currency values"""
    if amount >= 1000000:
        return f"{currency}{amount/1000000:.1f}M"
    elif amount >= 1000:
        return f"{currency}{amount/1000:.1f}K"
    else:
        return f"{currency}{amount:.1f}"

def format_points(points: float) -> str:
    """Format points display"""
    return f"{points:.1f} pts"

def calculate_form_trend(recent_points: List[float]) -> Dict:
    """Calculate form trend from recent points"""
    if len(recent_points) < 2:
        return {'trend': 'stable', 'direction': 0, 'consistency': 0}
    
    # Calculate trend
    x = np.arange(len(recent_points))
    slope = np.polyfit(x, recent_points, 1)[0]
    
    # Calculate consistency (inverse of variance)
    consistency = 1 / (np.var(recent_points) + 1e-6)
    
    trend = 'improving' if slope > 0.5 else 'declining' if slope < -0.5 else 'stable'
    
    return {
        'trend': trend,
        'direction': slope,
        'consistency': min(consistency, 10)  # Cap at 10 for display
    }

def get_player_summary_stats(player_data: Dict) -> Dict:
    """Generate summary statistics for a player"""
    stats = {
        'efficiency': player_data.get('total_points', 0) / max(player_data.get('now_cost', 40) / 10, 1),
        'reliability': min(player_data.get('minutes', 0) / 2500, 1),  # Based on max minutes
        'form_rating': float(player_data.get('form', 0)),
        'popularity': player_data.get('selected_by_percent', 0),
        'value_tier': 'premium' if player_data.get('now_cost', 0) >= 90 else 'mid' if player_data.get('now_cost', 0) >= 60 else 'budget'
    }
    
    # Overall rating (0-10 scale)
    stats['overall_rating'] = min(
        (stats['efficiency'] * 2 + stats['reliability'] * 2 + stats['form_rating']) / 5,
        10
    )
    
    return stats

# Initialize global instances
performance_tracker = PerformanceTracker()
cache_manager = CacheManager()
loggers = setup_logging()

# Load configuration
FPLConfig.load_custom_config()

import time