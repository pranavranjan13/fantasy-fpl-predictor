from pulp import *
from typing import List, Dict, Tuple
import pandas as pd
# Import Player, Position, TeamConstraints, TeamSelection from models.py
from models import Player, Position, TeamConstraints, TeamSelection

import pydantic
from pydantic import Field
print("PYDANTIC VERSION:", pydantic.__version__)
print("Field function:", Field)
print("Field doc:", Field.__doc__)

class TeamOptimizer:
    def __init__(self, constraints: TeamConstraints):
        self.constraints = constraints

    def optimize_team(self, players: List[Player], predictions: Dict[int, float]) -> TeamSelection:
        """Optimize team selection using linear programming"""
        # Create the problem
        prob = LpProblem("Fantasy_Team_Selection", LpMaximize)

        # Decision variables - binary for each player
        player_vars = {}
        for player in players:
            player_vars[player.id] = LpVariable(f"player_{player.id}", cat='Binary')

        # Objective function - maximize predicted points
        prob += lpSum([
            predictions.get(player.id, 0) * player_vars[player.id]
            for player in players
        ])

        # Budget constraint
        prob += lpSum([
            player.price * player_vars[player.id]
            for player in players
        ]) <= self.constraints.total_budget

        # Position constraints
        position_players = {}
        for pos in Position:
            position_players[pos] = [p for p in players if p.position == pos]
            # Exact number of players per position
            prob += lpSum([
                player_vars[player.id]
                for player in position_players[pos]
            ]) == self.constraints.required_positions[pos]

        # Team constraint - max 3 players from same team
        teams = set(player.team for player in players)
        for team in teams:
            team_players = [p for p in players if p.team == team]
            prob += lpSum([
                player_vars[player.id]
                for player in team_players
            ]) <= self.constraints.max_players_per_team

        # Solve the problem
        prob.solve(PULP_CBC_CMD(msg=0))

        # Extract solution
        selected_players = []
        total_cost = 0
        for player in players:
            if player_vars[player.id].varValue == 1:
                selected_players.append(player)
                total_cost += player.price

        # Select starting eleven and bench
        starting_eleven, bench = self._select_starting_eleven(selected_players, predictions)

        # Select captain (highest predicted points in starting eleven)
        captain_id = max(starting_eleven, key=lambda pid: predictions.get(pid, 0))
        vice_captain_candidates = [pid for pid in starting_eleven if pid != captain_id]
        vice_captain_id = max(vice_captain_candidates, key=lambda pid: predictions.get(pid, 0))

        total_predicted_points = sum(predictions.get(p.id, 0) for p in selected_players)

        return TeamSelection(
            players=selected_players,
            total_cost=total_cost,
            predicted_points=total_predicted_points,
            starting_eleven=starting_eleven,
            bench=bench,
            captain=captain_id,
            vice_captain=vice_captain_id
        )

    def _select_starting_eleven(self, players: List[Player], predictions: Dict[int, float]) -> Tuple[List[int], List[int]]:
        """Select starting eleven from the 15 selected players"""
        by_position = {}
        for pos in Position:
            by_position[pos] = [p for p in players if p.position == pos]
            # Sort by predicted points
            by_position[pos].sort(key=lambda p: predictions.get(p.id, 0), reverse=True)

        starting = []
        # Must start 1 GKP
        starting.extend([p.id for p in by_position[Position.GKP][:1]])
        # Must start at least 3 DEF, but can start up to 5
        starting.extend([p.id for p in by_position[Position.DEF][:3]])
        # Must start at least 2 MID, but can start up to 5  
        starting.extend([p.id for p in by_position[Position.MID][:3]])
        # Must start at least 1 FWD, but can start up to 3
        starting.extend([p.id for p in by_position[Position.FWD][:3]])

        # Fill remaining spots with highest predicted players
        remaining_players = [p for p in players if p.id not in starting]
        remaining_players.sort(key=lambda p: predictions.get(p.id, 0), reverse=True)

        while len(starting) < 11 and remaining_players:
            player = remaining_players.pop(0)
            # Check formation constraints
            pos_count = len([p for p in players if p.id in starting and p.position == player.position])
            max_pos = 5 if player.position in [Position.DEF, Position.MID] else (3 if player.position == Position.FWD else 1)
            if pos_count < max_pos:
                starting.append(player.id)
        bench = [p.id for p in players if p.id not in starting]
        return starting, bench