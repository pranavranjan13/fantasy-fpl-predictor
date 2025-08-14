import requests
import pandas as pd
import json
from typing import Dict, List
import asyncio
import httpx
# Import Player and Position from models.py
from backend.models import Player, Position

class FPLDataFetcher:
    def __init__(self):
        self.base_url = "https://fantasy.premierleague.com/api"
        self.session = requests.Session()

    def fetch_bootstrap_data(self) -> Dict:
        """Fetch general game data including players, teams, and game settings"""
        response = self.session.get(f"{self.base_url}/bootstrap-static/")
        return response.json()

    def fetch_player_data(self, player_id: int) -> Dict:
        """Fetch detailed data for a specific player"""
        response = self.session.get(f"{self.base_url}/element-summary/{player_id}/")
        return response.json()

    def fetch_fixtures(self) -> List[Dict]:
        """Fetch fixture data"""
        response = self.session.get(f"{self.base_url}/fixtures/")
        return response.json()

    def fetch_gameweek_data(self, gameweek: int) -> Dict:
        """Fetch live data for a specific gameweek"""
        response = self.session.get(f"{self.base_url}/event/{gameweek}/live/")
        return response.json()

    async def fetch_all_player_details(self, player_ids: List[int]) -> List[Dict]:
        """Fetch detailed data for multiple players asynchronously"""
        async with httpx.AsyncClient() as client:
            tasks = [
                client.get(f"{self.base_url}/element-summary/{player_id}/")
                for player_id in player_ids
            ]
            responses = await asyncio.gather(*tasks)
            return [response.json() for response in responses]

    def process_bootstrap_to_players(self, bootstrap_data: Dict) -> List[Player]:
        """Convert bootstrap data to Player objects"""
        players = []
        teams = {team['id']: team['name'] for team in bootstrap_data['teams']}
        positions = {pos['id']: pos['singular_name_short'] for pos in bootstrap_data['element_types']}
        for element in bootstrap_data['elements']:
            player = Player(
                id=element['id'],
                name=f"{element['first_name']} {element['second_name']}",
                position=Position(positions[element['element_type']]),
                team=teams[element['team']],
                price=element['now_cost'] / 10.0,
                total_points=element['total_points'],
                points_per_game=float(element['points_per_game']),
                selected_by_percent=float(element['selected_by_percent']),
                form=float(element['form']),
                minutes=element['minutes'],
                goals_scored=element['goals_scored'],
                assists=element['assists'],
                clean_sheets=element['clean_sheets'],
                goals_conceded=element['goals_conceded'],
                own_goals=element['own_goals'],
                penalties_saved=element['penalties_saved'],
                penalties_missed=element['penalties_missed'],
                yellow_cards=element['yellow_cards'],
                red_cards=element['red_cards'],
                saves=element['saves'],
                bonus=element['bonus'],
                bps=element['bps'],
                influence=float(element['influence']),
                creativity=float(element['creativity']),
                threat=float(element['threat']),
                ict_index=float(element['ict_index']),
                expected_goals=float(element.get('expected_goals', 0)),
                expected_assists=float(element.get('expected_assists', 0)),
                expected_goal_involvements=float(element.get('expected_goal_involvements', 0))
            )
            players.append(player)
        return players