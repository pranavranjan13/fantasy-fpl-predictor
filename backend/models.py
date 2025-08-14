from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class Position(str, Enum):
    GKP = "GKP"  # Goalkeeper
    DEF = "DEF"  # Defender
    MID = "MID"  # Midfielder
    FWD = "FWD"  # Forward

class ChipType(str, Enum):
    WILDCARD = "wildcard"
    FREE_HIT = "free_hit"
    BENCH_BOOST = "bench_boost"
    TRIPLE_CAPTAIN = "triple_captain"

class Player(BaseModel):
    id: int
    name: str
    position: Position
    team: str
    price: float
    total_points: int
    points_per_game: float
    selected_by_percent: float
    form: float
    minutes: int
    goals_scored: int
    assists: int
    clean_sheets: int
    goals_conceded: int
    own_goals: int
    penalties_saved: int
    penalties_missed: int
    yellow_cards: int
    red_cards: int
    saves: int
    bonus: int
    bps: int  # Bonus Points System
    influence: float
    creativity: float
    threat: float
    ict_index: float
    expected_goals: float
    expected_assists: float
    expected_goal_involvements: float

class TeamConstraints(BaseModel):
    total_budget: float = 100.0
    max_players_per_team: int = 3
    required_positions: Dict[Position, int] = {
        Position.GKP: 2,
        Position.DEF: 5,
        Position.MID: 5,
        Position.FWD: 3
    }
    starting_eleven_positions: Dict[Position, int] = {
        Position.GKP: 1,
        Position.DEF: 3,  # minimum
        Position.MID: 2,  # minimum
        Position.FWD: 1   # minimum
    }

class PredictionRequest(BaseModel):
    gameweek: int
    budget: float = 100.0
    constraints: Optional[TeamConstraints] = None
    current_team: Optional[List[int]] = None
    chips_used: Optional[List[ChipType]] = None

class TeamSelection(BaseModel):
    players: List[Player]
    total_cost: float
    predicted_points: float
    starting_eleven: List[int]
    bench: List[int]
    captain: int
    vice_captain: int

class ChipRecommendation(BaseModel):
    chip: ChipType
    recommended_gameweek: int
    confidence: float
    reasoning: str
    expected_benefit: float