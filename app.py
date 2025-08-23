import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="FPL AI Assistant",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #37003c;
        margin-bottom: 2rem;
    }
    .field-container {
        background: linear-gradient(135deg, #2e7d32 0%, #4caf50 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        position: relative;
        min-height: 500px;
    }
    .field-lines {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-image:
            linear-gradient(90deg, rgba(255,255,255,0.3) 1px, transparent 1px),
            linear-gradient(rgba(255,255,255,0.3) 1px, transparent 1px);
        background-size: 50px 50px;
        border-radius: 15px;
    }
    .player-position {
        position: absolute;
        background: white;
        border: 2px solid #37003c;
        border-radius: 50px;
        padding: 8px 12px;
        text-align: center;
        font-weight: bold;
        font-size: 12px;
        color: #37003c;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        min-width: 80px;
    }
    .captain-badge {
        background: #ffd700 !important;
        color: #000 !important;
        border-color: #ffd700 !important;
    }
    .vice-captain-badge {
        background: #c0c0c0 !important;
        color: #000 !important;
        border-color: #c0c0c0 !important;
    }
    .bench-container {
        background: #f5f5f5;
        border-radius: 10px;
        padding: 15px;
        margin-top: 20px;
    }
    .bench-player {
        background: #fff;
        border: 2px solid #666;
        border-radius: 25px;
        padding: 8px 15px;
        margin: 5px;
        display: inline-block;
        font-weight: bold;
        color: #333;
    }
    .chat-container {
        max-height: 400px;
        overflow-y: auto;
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        background: #f9f9f9;
    }
    .user-message {
        background: #e3f2fd;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        margin-left: 20px;
        text-align: right;
    }
    .ai-message {
        background: #f3e5f5;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        margin-right: 20px;
    }
</style>
""", unsafe_allow_html=True)


def safe_float(value, default=0.0):
    """Safely convert a value to float"""
    try:
        if pd.isna(value) or value == '' or value is None:
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


class EnhancedFPLDataManager:
    """Enhanced FPL data manager with historical analysis"""

    def __init__(self):
        self.base_url = "https://fantasy.premierleague.com/api/"
        self.current_season_data = None

    def fetch_current_season_data(self):
        """Fetch current season data from FPL API"""
        try:
            response = requests.get(
                f"{self.base_url}bootstrap-static/", timeout=15)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API returned status code: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"Error fetching FPL data: {str(e)}")
            return None

    def fetch_player_detailed_data(self, player_id):
        """Fetch detailed player data including fixtures and history"""
        try:
            response = requests.get(
                f"{self.base_url}element-summary/{player_id}/", timeout=10)
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None

    def get_players_dataframe(self):
        """Get enhanced player data with predictions"""
        if self.current_season_data is None:
            with st.spinner("Fetching FPL data..."):
                self.current_season_data = self.fetch_current_season_data()
        if self.current_season_data:
            players_df = pd.DataFrame(self.current_season_data['elements'])
            # Add position and team names
            position_map = {
                pos['id']: pos['singular_name'] for pos in self.current_season_data['element_types']}
            team_map = {team['id']: team['name']
                        for team in self.current_season_data['teams']}
            players_df['position'] = players_df['element_type'].map(
                position_map)
            players_df['team_name'] = players_df['team'].map(team_map)
            players_df['value'] = players_df['now_cost'].apply(
                lambda x: safe_float(x) / 10.0)
            players_df['form_float'] = players_df['form'].apply(safe_float)
            # Enhanced metrics
            players_df = self._calculate_enhanced_metrics(players_df)
            return players_df
        return pd.DataFrame()

    def _calculate_enhanced_metrics(self, df):
        """Calculate enhanced prediction metrics"""
        # Historical performance weight (based on last season equivalent)
        df['historical_weight'] = df.apply(
            self._calculate_historical_performance, axis=1)
        # Form trend (last 5 games weighted)
        df['form_trend'] = df['form_float'] * 1.5  # Weight recent form heavily
        # Fixture difficulty adjustment
        df['fixture_adjustment'] = df.apply(
            self._calculate_fixture_difficulty, axis=1)
        # Minutes reliability
        df['minutes_reliability'] = df['minutes'].apply(
            lambda x: min(safe_float(x) / 2500, 1.0))
        # ICT reliability
        df['ict_reliability'] = (
            df['influence'].apply(safe_float) * 0.4 +
            df['creativity'].apply(safe_float) * 0.3 +
            df['threat'].apply(safe_float) * 0.3
        ) / 100
        # Expected points for next gameweek
        df['predicted_points'] = df.apply(
            self._predict_next_gameweek_points, axis=1)
        # Value efficiency
        df['value_efficiency'] = df['predicted_points'] / \
            df['value'].apply(lambda x: max(x, 4.0))
        # Starting probability
        df['starting_probability'] = df.apply(
            self._calculate_starting_probability, axis=1)
        return df

    def _calculate_historical_performance(self, player):
        """Calculate historical performance weight"""
        total_points = safe_float(player.get('total_points', 0))
        games_played = max(safe_float(player.get('minutes', 0)) / 90, 1)
        # Historical consistency bonus
        ppg = total_points / games_played
        if ppg > 6:  # Premium performance
            return 1.3
        elif ppg > 4:  # Good performance
            return 1.1
        elif ppg > 2:  # Average performance
            return 1.0
        else:  # Below average
            return 0.8

    def _calculate_fixture_difficulty(self, player):
        """Calculate fixture difficulty adjustment"""
        # Simplified fixture difficulty - in real implementation would use actual fixtures
        team_strength = {
            'Manchester City': 1.3, 'Arsenal': 1.25, 'Liverpool': 1.25,
            'Chelsea': 1.15, 'Manchester United': 1.15, 'Tottenham': 1.1,
            'Newcastle United': 1.05, 'Brighton': 1.0, 'Aston Villa': 1.0
        }
        team_name = player.get('team_name', '')
        return team_strength.get(team_name, 0.95)  # Default slightly below average

    def _predict_next_gameweek_points(self, player):
        """Predict points for next gameweek using multiple factors"""
        base_points = safe_float(player.get('total_points', 0))
        games_played = max(safe_float(player.get('minutes', 0)) / 90, 1)
        ppg = base_points / games_played
        # Factor weights
        form_weight = 0.35
        historical_weight = 0.25
        fixture_weight = 0.20
        ict_weight = 0.15
        minutes_weight = 0.05
        # Calculate prediction
        prediction = (
            ppg * player.get('historical_weight', 1.0) * historical_weight +
            player.get('form_trend', 0) * form_weight +
            player.get('fixture_adjustment', 1.0) * fixture_weight * ppg +
            player.get('ict_reliability', 0) * ict_weight +
            player.get('minutes_reliability', 0) * minutes_weight * ppg
        )
        # Position adjustments
        position = player.get('position', '')
        position_multipliers = {
            'Goalkeeper': 0.7,
            'Defender': 0.8,
            'Midfielder': 1.2,
            'Forward': 1.3
        }
        return prediction * position_multipliers.get(position, 1.0)

    def _calculate_starting_probability(self, player):
        """Calculate probability of starting next game"""
        minutes = safe_float(player.get('minutes', 0))
        total_games = 25  # Approximate games so far
        if minutes > (total_games * 70):  # Plays most games, most minutes
            return 0.95
        elif minutes > (total_games * 50):  # Regular starter
            return 0.85
        elif minutes > (total_games * 30):  # Squad rotation player
            return 0.65
        elif minutes > (total_games * 10):  # Occasional starter
            return 0.40
        else:  # Rarely plays
            return 0.15


class EnhancedFPLPredictor:
    """Enhanced predictor with better team selection logic"""

    def __init__(self, players_df):
        self.players_df = players_df
        self.position_limits = {"Goalkeeper": 2,
                                "Defender": 5, "Midfielder": 5, "Forward": 3}

    def optimize_team_selection(self, budget=100.0):
        """Generate optimal 15-player squad using enhanced logic"""
        if self.players_df.empty:
            return {"error": "No player data available"}
        selected_players = []
        remaining_budget = budget
        team_counts = {}  # Track players per team (max 3)
        # Sort all players by value efficiency
        efficient_players = self.players_df.sort_values(
            'value_efficiency', ascending=False)
        # Selection strategy: fill positions with best value players
        for position in ["Goalkeeper", "Defender", "Midfielder", "Forward"]:
            limit = self.position_limits[position]
            pos_players = efficient_players[efficient_players['position']
                                            == position].copy()
            position_selected = []
            for _, player in pos_players.iterrows():
                if len(position_selected) >= limit:
                    break
                player_cost = player['value']
                player_team = player['team']
                # Check budget constraint
                if player_cost > remaining_budget:
                    continue
                # Check team limit (max 3 per team)
                if team_counts.get(player_team, 0) >= 3:
                    continue
                # Add player
                position_selected.append(player.to_dict())
                remaining_budget -= player_cost
                team_counts[player_team] = team_counts.get(player_team, 0) + 1
            selected_players.extend(position_selected)
        # Fill remaining spots if budget allows and positions not full
        if len(selected_players) < 15 and remaining_budget > 4.0:
            remaining_players = efficient_players[
                ~efficient_players.index.isin(
                    [p['id'] if 'id' in p else i for i, p in enumerate(selected_players)])
            ]
            for _, player in remaining_players.iterrows():
                if len(selected_players) >= 15:
                    break
                player_cost = player['value']
                player_team = player['team']
                position = player['position']
                # Count current position players
                pos_count = sum(
                    1 for p in selected_players if p.get('position') == position)
                if (player_cost <= remaining_budget and
                    team_counts.get(player_team, 0) < 3 and
                        pos_count < self.position_limits.get(position, 0)):
                    selected_players.append(player.to_dict())
                    remaining_budget -= player_cost
                    team_counts[player_team] = team_counts.get(
                        player_team, 0) + 1
        total_cost = sum(p['value'] for p in selected_players)
        return {
            "team": selected_players,
            "total_cost": total_cost,
            "remaining_budget": budget - total_cost,
            "team_distribution": team_counts
        }

    def suggest_starting_eleven(self, team_players):
        """Suggest optimal starting XI using advanced formation logic"""
        if isinstance(team_players, list):
            team_df = pd.DataFrame(team_players)
        else:
            team_df = team_players.copy()
        if team_df.empty:
            return {"error": "No team data available"}
        # Try multiple formations and pick best
        formations = [
            {"name": "3-5-2", "GK": 1, "DEF": 3, "MID": 5, "FWD": 2},
            {"name": "3-4-3", "GK": 1, "DEF": 3, "MID": 4, "FWD": 3},
            {"name": "4-4-2", "GK": 1, "DEF": 4, "MID": 4, "FWD": 2},
            {"name": "4-3-3", "GK": 1, "DEF": 4, "MID": 3, "FWD": 3},
            {"name": "5-3-2", "GK": 1, "DEF": 5, "MID": 3, "FWD": 2}
        ]
        best_formation = None
        best_score = -1
        for formation in formations:
            xi_players = []
            total_predicted = 0
            # Select best players for each position
            for pos_name, count in [("Goalkeeper", formation["GK"]),
                                    ("Defender", formation["DEF"]),
                                    ("Midfielder", formation["MID"]),
                                    ("Forward", formation["FWD"])]:
                pos_players = team_df[team_df['position'] == pos_name]
                if len(pos_players) < count:
                    break  # Can't form this formation
                # Select best players considering both predicted points and starting probability
                pos_players['selection_score'] = (
                    pos_players['predicted_points'] * 0.7 +
                    pos_players['starting_probability'] *
                    pos_players['predicted_points'] * 0.3
                )
                best_pos_players = pos_players.nlargest(count, 'selection_score')
                xi_players.extend(best_pos_players.to_dict('records'))
                total_predicted += best_pos_players['predicted_points'].sum()
            if len(xi_players) == 11 and total_predicted > best_score:
                # Find captain and vice-captain
                xi_df = pd.DataFrame(xi_players)
                captain_idx = xi_df['predicted_points'].idxmax()
                captain = xi_df.loc[captain_idx].to_dict()
                vice_candidates = xi_df.drop(captain_idx)
                vice_captain_idx = vice_candidates['predicted_points'].idxmax()
                vice_captain = vice_candidates.loc[vice_captain_idx].to_dict()
                # Find bench players
                selected_ids = set(
                    p.get('id', p.get('web_name')) for p in xi_players)
                bench_players = []
                for _, player in team_df.iterrows():
                    player_id = player.get('id', player.get('web_name'))
                    if player_id not in selected_ids:
                        bench_players.append(player.to_dict())
                best_formation = {
                    "formation": formation,
                    "players": xi_players,
                    "total_predicted_points": total_predicted,
                    "captain": captain,
                    "vice_captain": vice_captain,
                    "bench": bench_players[:4]  # Only show first 4 bench players
                }
                best_score = total_predicted
        return best_formation if best_formation else {"error": "Could not form valid XI"}


class EnhancedFPLChatBot:
    """Enhanced chatbot with AI API integration"""

    def __init__(self, api_key=None):
        self.api_key = api_key
        self.chat_history = []
        self.fpl_knowledge = {
            "captain_tips": "Choose captains based on: 1) Favorable fixtures 2) Recent form 3) Historical performance against opponent 4) Injury status 5) Expected minutes",
            "transfer_strategy": "Transfer tips: 1) Don't rush early transfers 2) Plan 2-3 gameweeks ahead 3) Consider price changes 4) Bank transfers for double gameweeks 5) Avoid taking hits unless essential",
            "formation_guide": "Formation selection: 3-5-2 for premium midfielders, 3-4-3 for attacking returns, 4-4-2 for balance, 4-3-3 for strong defense, 5-3-2 for premium defenders",
            "chip_timing": "Chip strategy: Wildcard 1 (GW8-12), Triple Captain (double gameweeks), Bench Boost (full team has fixtures), Free Hit (blank gameweeks), Wildcard 2 (GW20-25)"
        }

    def get_ai_response(self, message, context=None):
        """Get AI response - try API first, then fallback"""
        if self.api_key:
            try:
                return self._get_api_response(message, context)
            except Exception as e:
                st.warning(f"AI API unavailable, using knowledge base response")
                return self._get_knowledge_response(message)
        else:
            return self._get_knowledge_response(message)

    def _get_api_response(self, message, context):
        """Get response from AI API"""
        system_prompt = """You are an expert Fantasy Premier League (FPL) advisor with deep knowledge of:
        - Player performance analysis and statistics
        - Transfer strategies and market timing  
        - Captain selection and differential picks
        - Formation tactics and team selection
        - Chip usage strategy and timing
        - Fixture analysis and season planning
        Provide specific, actionable FPL advice. Be concise but comprehensive.
        Focus on data-driven insights and practical recommendations."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        messages = [{"role": "system", "content": system_prompt}]
        # Add recent chat history for context
        if self.chat_history:
            messages.extend(self.chat_history[-4:])
        messages.append({"role": "user", "content": message})
        # Try multiple API endpoints
        api_endpoints = [
            "https://api.anthropic.com/v1/messages",
            "https://api.openai.com/v1/chat/completions"
        ]
        for endpoint in api_endpoints:
            try:
                response = requests.post(endpoint, headers=headers, json={
                    "model": "claude-3-sonnet-20240229" if "anthropic" in endpoint else "gpt-3.5-turbo",
                    "messages": messages,
                    "max_tokens": 500,
                    "temperature": 0.7
                }, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if "anthropic" in endpoint:
                        return data.get('content', [{}])[0].get('text', '')
                    else:
                        return data.get('choices', [{}])[0].get('message', {}).get('content', '')
            except:
                continue
        raise Exception("All API endpoints failed")

    def _get_knowledge_response(self, message):
        """Get response from knowledge base"""
        message_lower = message.lower()
        if any(word in message_lower for word in ['captain', 'c', 'armband', 'who to captain']):
            return f"🎯 **Captain Selection Strategy:**\n\n{self.fpl_knowledge['captain_tips']}\n\n**This Week's Top Picks:** Look for players with home fixtures against weaker opponents, good recent form (5+ points in last 3 games), and high expected minutes (>75%)."
        elif any(word in message_lower for word in ['transfer', 'buy', 'sell', 'move', 'swap']):
            return f"🔄 **Transfer Strategy Guide:**\n\n{self.fpl_knowledge['transfer_strategy']}\n\n**Current Focus:** Identify players with favorable fixture swings over the next 3-4 gameweeks. Avoid sideways transfers unless for significant fixture improvement."
        elif any(word in message_lower for word in ['formation', 'starting', 'xi', 'lineup', '11']):
            return f"⚡ **Formation Selection Guide:**\n\n{self.fpl_knowledge['formation_guide']}\n\n**Quick Tips:** Match your formation to your premium players' positions. If you have 3 premium midfielders, use 3-5-2. If you have premium forwards, consider 3-4-3."
        elif any(word in message_lower for word in ['chip', 'wildcard', 'triple', 'bench boost', 'free hit']):
            return f"💎 **Chip Usage Strategy:**\n\n{self.fpl_knowledge['chip_timing']}\n\n**Current Recommendations:** Plan your chips around double and blank gameweeks. Use data from FPL websites to time chip usage with favorable fixtures for your players."
        elif any(word in message_lower for word in ['differential', 'punt', 'unique', 'template']):
            return "🎯 **Differential Strategy:**\n\n• Target players with <10% ownership who have good fixtures\n• Focus on attacking players from mid-table teams\n• Avoid differential defenders unless exceptional fixtures\n• Time differentials with favorable fixture runs\n• Limit to 2-3 differentials max to manage risk"
        elif any(word in message_lower for word in ['budget', 'money', 'price', 'value', 'cost']):
            return "💰 **Budget Management:**\n\n• Spend 65-70% on outfield players\n• Don't overspend on goalkeepers (4.5M max recommended)\n• Invest in 2-3 premium players (8M+)\n• Use 4.0-4.5M defender enablers for rotation\n• Keep 0.5-1.0M in the bank for flexibility\n• Monitor price changes on FPL price change websites"
        else:
            return "🏆 **General FPL Strategy:**\n\n• **Research:** Check team news, injury reports, and press conferences\n• **Planning:** Analyze fixtures 3-4 gameweeks ahead\n• **Patience:** Avoid knee-jerk reactions to single gameweeks\n• **Data:** Use underlying stats (xG, xA) from sites like Understat\n• **Community:** Follow FPL Twitter and Reddit for insights\n• **Consistency:** Stick to your strategy but be flexible with form changes"

    def add_to_history(self, user_msg, ai_response):
        """Add interaction to chat history"""
        self.chat_history.append({"role": "user", "content": user_msg})
        self.chat_history.append(
            {"role": "assistant", "content": ai_response})
        # Keep only last 10 interactions to manage memory
        if len(self.chat_history) > 20:
            self.chat_history = self.chat_history[-20:]


def create_field_visualization(xi_players, formation, captain, vice_captain, bench_players):
    """Create football field visualization using streamlit native components"""
    st.markdown("### 🏟️ Formation Layout")
    # Group players by position for display
    players_by_pos = {
        'Goalkeeper': [],
        'Defender': [],
        'Midfielder': [],
        'Forward': []
    }
    for player in xi_players:
        pos = player.get('position', 'Unknown')
        if pos in players_by_pos:
            players_by_pos[pos].append(player)
    # Display formation info
    st.info(
        f"**Formation:** {formation['name']} | **Total Predicted Points:** {sum(safe_float(p.get('predicted_points', 0)) for p in xi_players):.1f}")
    # Display players by position with better visibility
    st.markdown("#### ⚽ **FORWARDS**")
    if players_by_pos['Forward']:
        fwd_cols = st.columns(len(players_by_pos['Forward']))
        for i, player in enumerate(players_by_pos['Forward']):
            with fwd_cols[i]:
                badge = ""
                bg_color = "#ffffff"
                text_color = "#000000"  # Always black for visibility
                border_color = "#37003c"
                if player['web_name'] == captain['web_name']:
                    badge = "👑 (C) "
                    bg_color = "#ffd700"
                    border_color = "#b8860b"
                elif player['web_name'] == vice_captain['web_name']:
                    badge = "🥈 (VC) "
                    bg_color = "#c0c0c0"
                    border_color = "#808080"
                st.markdown(f"""
                <div style="
                    background-color: {bg_color};
                    color: {text_color};
                    border: 3px solid {border_color};
                    border-radius: 20px;
                    padding: 20px 15px;
                    text-align: center;
                    font-weight: bold;
                    margin: 10px 5px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                    min-height: 100px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                ">
                    <div style="font-size: 16px; color: {text_color}; margin-bottom: 5px;">
                        {badge}
                    </div>
                    <div style="font-size: 14px; font-weight: bold; color: {text_color}; margin-bottom: 8px;">
                        {player['web_name']}
                    </div>
                    <div style="font-size: 12px; color: #2c5aa0; font-weight: bold;">
                        {safe_float(player.get('predicted_points', 0)):.1f} pts
                    </div>
                    <div style="font-size: 10px; color: #28a745; margin-top: 3px;">
                        £{safe_float(player.get('value', 0)):.1f}M
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.write("No forwards selected")
    st.markdown("#### 🎯 **MIDFIELDERS**")
    if players_by_pos['Midfielder']:
        mid_cols = st.columns(len(players_by_pos['Midfielder']))
        for i, player in enumerate(players_by_pos['Midfielder']):
            with mid_cols[i]:
                badge = ""
                bg_color = "#ffffff"
                text_color = "#000000"  # Always black for visibility
                border_color = "#37003c"
                if player['web_name'] == captain['web_name']:
                    badge = "👑 (C) "
                    bg_color = "#ffd700"
                    border_color = "#b8860b"
                elif player['web_name'] == vice_captain['web_name']:
                    badge = "🥈 (VC) "
                    bg_color = "#c0c0c0"
                    border_color = "#808080"
                st.markdown(f"""
                <div style="
                    background-color: {bg_color};
                    color: {text_color};
                    border: 3px solid {border_color};
                    border-radius: 20px;
                    padding: 20px 15px;
                    text-align: center;
                    font-weight: bold;
                    margin: 10px 5px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                    min-height: 100px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                ">
                    <div style="font-size: 16px; color: {text_color}; margin-bottom: 5px;">
                        {badge}
                    </div>
                    <div style="font-size: 14px; font-weight: bold; color: {text_color}; margin-bottom: 8px;">
                        {player['web_name']}
                    </div>
                    <div style="font-size: 12px; color: #2c5aa0; font-weight: bold;">
                        {safe_float(player.get('predicted_points', 0)):.1f} pts
                    </div>
                    <div style="font-size: 10px; color: #28a745; margin-top: 3px;">
                        £{safe_float(player.get('value', 0)):.1f}M
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.write("No midfielders selected")
    st.markdown("#### 🛡️ **DEFENDERS**")
    if players_by_pos['Defender']:
        def_cols = st.columns(len(players_by_pos['Defender']))
        for i, player in enumerate(players_by_pos['Defender']):
            with def_cols[i]:
                badge = ""
                bg_color = "#ffffff"
                text_color = "#000000"  # Always black for visibility
                border_color = "#37003c"
                if player['web_name'] == captain['web_name']:
                    badge = "👑 (C) "
                    bg_color = "#ffd700"
                    border_color = "#b8860b"
                elif player['web_name'] == vice_captain['web_name']:
                    badge = "🥈 (VC) "
                    bg_color = "#c0c0c0"
                    border_color = "#808080"
                st.markdown(f"""
                <div style="
                    background-color: {bg_color};
                    color: {text_color};
                    border: 3px solid {border_color};
                    border-radius: 20px;
                    padding: 20px 15px;
                    text-align: center;
                    font-weight: bold;
                    margin: 10px 5px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                    min-height: 100px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                ">
                    <div style="font-size: 16px; color: {text_color}; margin-bottom: 5px;">
                        {badge}
                    </div>
                    <div style="font-size: 14px; font-weight: bold; color: {text_color}; margin-bottom: 8px;">
                        {player['web_name']}
                    </div>
                    <div style="font-size: 12px; color: #2c5aa0; font-weight: bold;">
                        {safe_float(player.get('predicted_points', 0)):.1f} pts
                    </div>
                    <div style="font-size: 10px; color: #28a745; margin-top: 3px;">
                        £{safe_float(player.get('value', 0)):.1f}M
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.write("No defenders selected")
    st.markdown("#### 🥅 **GOALKEEPER**")
    if players_by_pos['Goalkeeper']:
        # Center the goalkeeper
        _, gk_col, _ = st.columns([1, 1, 1])
        with gk_col:
            player = players_by_pos['Goalkeeper'][0]
            badge = ""
            bg_color = "#ffffff"
            text_color = "#000000"  # Always black for visibility
            border_color = "#37003c"
            if player['web_name'] == captain['web_name']:
                badge = "👑 (C) "
                bg_color = "#ffd700"
                border_color = "#b8860b"
            elif player['web_name'] == vice_captain['web_name']:
                badge = "🥈 (VC) "
                bg_color = "#c0c0c0"
                border_color = "#808080"
            st.markdown(f"""
            <div style="
                background-color: {bg_color};
                color: {text_color};
                border: 3px solid {border_color};
                border-radius: 20px;
                padding: 20px 15px;
                text-align: center;
                font-weight: bold;
                margin: 10px 5px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                min-height: 100px;
                display: flex;
                flex-direction: column;
                justify-content: center;
            ">
                <div style="font-size: 16px; color: {text_color}; margin-bottom: 5px;">
                    {badge}
                </div>
                <div style="font-size: 14px; font-weight: bold; color: {text_color}; margin-bottom: 8px;">
                    {player['web_name']}
                </div>
                <div style="font-size: 12px; color: #2c5aa0; font-weight: bold;">
                    {safe_float(player.get('predicted_points', 0)):.1f} pts
                </div>
                <div style="font-size: 10px; color: #28a745; margin-top: 3px;">
                    £{safe_float(player.get('value', 0)):.1f}M
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.write("No goalkeeper selected")
    # Display bench with better formatting and visibility
    st.markdown("### 🪑 **Bench Players**")
    if bench_players and len(bench_players) > 0:
        # Create 4 columns for bench players
        bench_cols = st.columns(4)
        for i in range(min(4, len(bench_players))):
            player = bench_players[i]
            with bench_cols[i]:
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #f8f9fa, #e9ecef);
                    border: 3px solid #495057;
                    border-radius: 15px;
                    padding: 20px 10px;
                    text-align: center;
                    margin: 10px 0;
                    box-shadow: 0 3px 6px rgba(0,0,0,0.2);
                    min-height: 140px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                ">
                    <div style="font-size: 20px; font-weight: bold; color: #000000; margin-bottom: 5px;">
                        {i+1}
                    </div>
                    <div style="font-weight: bold; color: #000000; margin-bottom: 8px; font-size: 14px;">
                        {player.get('web_name', 'Unknown')}
                    </div>
                    <div style="color: #6c757d; font-size: 12px; margin: 3px 0; font-weight: bold;">
                        {player.get('position', 'Unknown')}
                    </div>
                    <div style="color: #28a745; font-weight: bold; font-size: 12px; margin: 2px 0;">
                        £{safe_float(player.get('value', 0)):.1f}M
                    </div>
                    <div style="color: #007bff; font-weight: bold; font-size: 12px;">
                        {safe_float(player.get('predicted_points', 0)):.1f} pts
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No bench players available")


def create_alternative_field_viz(xi_players, formation, captain, vice_captain, bench_players):
    """Create interactive field visualization using Plotly"""
    # Position coordinates for formations (normalized 0-1)
    positions = {
        "3-5-2": {
            "Goalkeeper": [(0.5, 0.1)],
            "Defender": [(0.2, 0.3), (0.5, 0.3), (0.8, 0.3)],
            "Midfielder": [(0.15, 0.6), (0.35, 0.5), (0.5, 0.7), (0.65, 0.5), (0.85, 0.6)],
            "Forward": [(0.35, 0.9), (0.65, 0.9)]
        },
        "3-4-3": {
            "Goalkeeper": [(0.5, 0.1)],
            "Defender": [(0.2, 0.3), (0.5, 0.3), (0.8, 0.3)],
            "Midfielder": [(0.25, 0.55), (0.42, 0.6), (0.58, 0.6), (0.75, 0.55)],
            "Forward": [(0.2, 0.85), (0.5, 0.9), (0.8, 0.85)]
        },
        "4-4-2": {
            "Goalkeeper": [(0.5, 0.1)],
            "Defender": [(0.15, 0.3), (0.35, 0.25), (0.65, 0.25), (0.85, 0.3)],
            "Midfielder": [(0.2, 0.55), (0.4, 0.6), (0.6, 0.6), (0.8, 0.55)],
            "Forward": [(0.35, 0.85), (0.65, 0.85)]
        },
        "4-3-3": {
            "Goalkeeper": [(0.5, 0.1)],
            "Defender": [(0.15, 0.3), (0.35, 0.25), (0.65, 0.25), (0.85, 0.3)],
            "Midfielder": [(0.3, 0.55), (0.5, 0.6), (0.7, 0.55)],
            "Forward": [(0.2, 0.85), (0.5, 0.9), (0.8, 0.85)]
        },
        "5-3-2": {
            "Goalkeeper": [(0.5, 0.1)],
            "Defender": [(0.1, 0.35), (0.25, 0.25), (0.5, 0.3), (0.75, 0.25), (0.9, 0.35)],
            "Midfielder": [(0.3, 0.55), (0.5, 0.6), (0.7, 0.55)],
            "Forward": [(0.35, 0.85), (0.65, 0.85)]
        }
    }
    formation_name = formation["name"]
    coords = positions.get(formation_name, positions["3-4-3"])
    fig = go.Figure()
    # Add field background (green rectangle)
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=1, y1=1,
        fillcolor="rgba(76, 175, 80, 0.8)",
        line=dict(color="white", width=3)
    )
    # Add field markings
    # Center line
    fig.add_shape(type="line", x0=0, y0=0.5, x1=1, y1=0.5,
                  line=dict(color="white", width=2))
    # Center circle
    fig.add_shape(type="circle", x0=0.4, y0=0.4, x1=0.6, y1=0.6,
                  line=dict(color="white", width=2))
    # Goal areas
    fig.add_shape(type="rect", x0=0.3, y0=0, x1=0.7, y1=0.15,
                  line=dict(color="white", width=2), fillcolor="rgba(255,255,255,0.1)")
    fig.add_shape(type="rect", x0=0.3, y0=0.85, x1=0.7, y1=1,
                  line=dict(color="white", width=2), fillcolor="rgba(255,255,255,0.1)")
    # Group players by position
    players_by_pos = {}
    for player in xi_players:
        pos = player['position']
        if pos not in players_by_pos:
            players_by_pos[pos] = []
        players_by_pos[pos].append(player)
    # Add players to the plot with proper spacing
    for position, pos_coords in coords.items():
        if position in players_by_pos:
            players = players_by_pos[position]
            for i, player in enumerate(players[:len(pos_coords)]):
                x, y = pos_coords[i]
                # Determine marker style
                if player['web_name'] == captain['web_name']:
                    color = '#ffd700'
                    symbol = 'star'
                    size = 30
                    name_suffix = " (C)"
                elif player['web_name'] == vice_captain['web_name']:
                    color = '#c0c0c0'
                    symbol = 'diamond'
                    size = 25
                    name_suffix = " (VC)"
                else:
                    color = 'white'
                    symbol = 'circle'
                    size = 20
                    name_suffix = ""
                # Add player marker
                fig.add_trace(go.Scatter(
                    x=[x], y=[y],
                    mode='markers+text',
                    marker=dict(
                        color=color,
                        size=size,
                        symbol=symbol,
                        line=dict(color='#37003c', width=2)
                    ),
                    text=f"{player['web_name']}{name_suffix}",
                    textposition="bottom center",
                    textfont=dict(size=10, color='white'),
                    showlegend=False,
                    hoverinfo='text',
                    hovertext=f"""
                    <b>{player['web_name']}</b><br>
                    Position: {player['position']}<br>
                    Price: £{safe_float(player.get('value', 0)):.1f}M<br>
                    Predicted Points: {safe_float(player.get('predicted_points', 0)):.1f}
                    """
                ))
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Formation: {formation_name}",
            x=0.5,
            font=dict(size=20, color='white')
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-0.1, 1.1]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-0.1, 1.1],
            scaleanchor="x",
            scaleratio=1
        ),
        plot_bgcolor='rgba(76, 175, 80, 0.9)',
        paper_bgcolor='rgba(46, 125, 50, 0.9)',
        height=600,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig


def main():
    """Main application"""
    st.markdown(
        '<h1 class="main-header">⚽ Enhanced FPL AI Assistant</h1>', unsafe_allow_html=True)
    # Initialize components
    data_manager = EnhancedFPLDataManager()
    # Sidebar for API key and navigation
    st.sidebar.title("🔧 Settings")
    api_key = st.sidebar.text_input("AI API Key (Optional)", type="password",
                                    help="Enter OpenAI or Anthropic API key for enhanced AI responses")
    chatbot = EnhancedFPLChatBot(api_key)
    if api_key:
        st.sidebar.success("✅ AI API configured")
    else:
        st.sidebar.info("💡 Add API key for enhanced AI chat")
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "Team Optimization",
        "Squad Analysis",
        "FPL Strategy Chat",
        "Player Comparison"
    ])
    # Load data
    players_df = data_manager.get_players_dataframe()
    if players_df.empty:
        st.error(
            "❌ Unable to load FPL data. Please check your internet connection and try again.")
        st.info(
            "💡 The app needs to connect to the official FPL API to fetch current player data.")
        return
    predictor = EnhancedFPLPredictor(players_df)
    st.success(
        f"✅ Loaded enhanced data for {len(players_df)} players with ML predictions")
    if page == "Team Optimization":
        st.header("🎯 Advanced Team Optimization")
        col1, col2 = st.columns([3, 1])
        with col1:
            budget = st.slider("Budget (£M)", 95.0, 105.0, 100.0, 0.5)
            if st.button("🚀 Generate Optimal Team with Field View", type="primary"):
                with st.spinner("Running advanced optimization..."):
                    result = predictor.optimize_team_selection(budget)
                    if "error" in result:
                        st.error(f"❌ {result['error']}")
                    else:
                        st.success(
                            f"✅ Generated optimal team! Cost: £{result['total_cost']:.1f}M (Remaining: £{result['remaining_budget']:.1f}M)")
                        # Get starting XI
                        xi_result = predictor.suggest_starting_eleven(
                            result['team'])
                        if "error" not in xi_result:
                            formation = xi_result['formation']
                            captain = xi_result['captain']
                            vice_captain = xi_result['vice_captain']
                            bench_players = xi_result['bench']
                            # Display formation and predicted points
                            col_form, col_points = st.columns(2)
                            with col_form:
                                st.info(f"🏟️ **Formation:** {formation['name']}")
                            with col_points:
                                st.info(
                                    f"📊 **Predicted XI Points:** {xi_result['total_predicted_points']:.1f}")
                            # Captain info
                            col_c1, col_c2 = st.columns(2)
                            with col_c1:
                                st.success(
                                    f"👑 **Captain:** {captain['web_name']} ({captain['predicted_points']:.1f} pts)")
                            with col_c2:
                                st.success(
                                    f"🎖️ **Vice-Captain:** {vice_captain['web_name']} ({vice_captain['predicted_points']:.1f} pts)")
                            # Field visualization - Fix the reset issue by storing result in session state
                            st.subheader("🏟️ Team Formation Visualization")
                            # Store the XI result in session state to prevent reset
                            if 'current_xi_result' not in st.session_state or st.session_state.get('regenerate_team', False):
                                st.session_state.current_xi_result = xi_result
                                st.session_state.regenerate_team = False
                            # Use stored result to prevent reset
                            stored_xi = st.session_state.current_xi_result
                            stored_formation = stored_xi['formation']
                            stored_captain = stored_xi['captain']
                            stored_vice_captain = stored_xi['vice_captain']
                            stored_bench_players = stored_xi.get('bench', [])
                            # Create visualization options without resetting data
                            viz_option = st.radio(
                                "Choose visualization:",
                                ["Interactive Field", "Simple Layout"],
                                horizontal=True,
                                key="team_viz_option"
                            )
                            if viz_option == "Interactive Field":
                                # Use Plotly for interactive visualization
                                fig = create_alternative_field_viz(
                                    stored_xi['players'], stored_formation, stored_captain,
                                    stored_vice_captain, stored_bench_players
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                # Show bench separately for interactive mode
                                st.markdown("### 🪑 **Bench Players**")
                                if stored_bench_players:
                                    bench_cols = st.columns(4)
                                    for i in range(min(4, len(stored_bench_players))):
                                        player = stored_bench_players[i]
                                        with bench_cols[i]:
                                            st.markdown(f"""
                                            <div style="
                                                background: linear-gradient(135deg, #f8f9fa, #e9ecef);
                                                border: 3px solid #495057;
                                                border-radius: 15px;
                                                padding: 15px 10px;
                                                text-align: center;
                                                margin: 5px;
                                                box-shadow: 0 3px 6px rgba(0,0,0,0.2);
                                                min-height: 120px;
                                            ">
                                                <div style="font-size: 18px; font-weight: bold; color: #000000;">
                                                    {i+1}
                                                </div>
                                                <div style="font-weight: bold; color: #000000; margin: 8px 0; font-size: 13px;">
                                                    {player.get('web_name', 'Unknown')}
                                                </div>
                                                <div style="color: #495057; font-size: 11px; font-weight: bold;">
                                                    {player.get('position', 'Unknown')}
                                                </div>
                                                <div style="color: #28a745; font-weight: bold; font-size: 11px;">
                                                    £{safe_float(player.get('value', 0)):.1f}M
                                                </div>
                                                <div style="color: #007bff; font-weight: bold; font-size: 11px;">
                                                    {safe_float(player.get('predicted_points', 0)):.1f} pts
                                                </div>
                                            </div>
                                            """, unsafe_allow_html=True)
                                else:
                                    st.info("No bench players available")
                            else:
                                # Use the improved simple layout
                                create_field_visualization(
                                    stored_xi['players'], stored_formation, stored_captain,
                                    stored_vice_captain, stored_bench_players
                                )
                            # Detailed squad analysis
                            with st.expander("📊 Detailed Squad Analysis"):
                                squad_df = pd.DataFrame(result['team'])
                                # Key statistics
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric(
                                        "Squad Value", f"£{result['total_cost']:.1f}M")
                                with col2:
                                    st.metric(
                                        "Avg Predicted Points", f"{squad_df['predicted_points'].mean():.1f}")
                                with col3:
                                    st.metric("High Starters", f"{sum(1 for p in result['team'] if p['starting_probability'] > 0.8)}")
                                with col4:
                                    st.metric(
                                        "Teams Used", len(result['team_distribution']))
                                # Position breakdown with enhanced stats
                                st.subheader("Position Analysis")
                                for pos in ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']:
                                    pos_players = [
                                        p for p in result['team'] if p['position'] == pos]
                                    if pos_players:
                                        st.write(
                                            f"**{pos}s ({len(pos_players)}):**")
                                        for player in sorted(pos_players, key=lambda x: x['predicted_points'], reverse=True):
                                            col_name, col_price, col_pred, col_prob, col_eff = st.columns([
                                                                                                           3, 1, 1, 1, 1])
                                            with col_name:
                                                st.write(
                                                    f"• **{player['web_name']}** ({player['team_name']})")
                                            with col_price:
                                                st.write(
                                                    f"£{player['value']:.1f}M")
                                            with col_pred:
                                                st.write(
                                                    f"{player['predicted_points']:.1f} pts")
                                            with col_prob:
                                                st.write(
                                                    f"{player['starting_probability']:.0%}")
                                            with col_eff:
                                                st.write(
                                                    f"{player['value_efficiency']:.1f}")
                                        st.write("")
                        else:
                            st.error(
                                "❌ Could not generate starting XI from optimal team")
        with col2:
            st.subheader("🔍 Optimization Info")
            if not players_df.empty:
                st.info(f"""
                **Enhanced Features:**
                • Historical performance analysis
                • Form trend weighting  
                • Fixture difficulty adjustment
                • Starting probability calculation
                • Value efficiency optimization
                • Team balance constraints
                """)
                # Top value picks
                st.subheader("💎 Top Value Picks")
                top_value = players_df.nlargest(5, 'value_efficiency')[
                    ['web_name', 'position', 'value', 'value_efficiency']]
                for _, player in top_value.iterrows():
                    st.write(
                        f"**{player['web_name']}** ({player['position']}) - {player['value_efficiency']:.2f}")
                # Form players
                st.subheader("🔥 In-Form Players")
                top_form = players_df.nlargest(5, 'form_trend')[
                    ['web_name', 'position', 'form_trend']]
                for _, player in top_form.iterrows():
                    st.write(
                        f"**{player['web_name']}** ({player['position']}) - {player['form_trend']:.1f}")
    elif page == "Squad Analysis":
        st.header("📋 Enhanced Squad Analysis")
        st.info(
            "🔍 Select your current 15 players for advanced AI analysis and optimization")
        # Enhanced player selection with search
        current_squad = []
        with st.expander("🔧 Select Your Current Squad", expanded=True):
            cols = st.columns(3)
            player_names = [''] + sorted(players_df['web_name'].unique())
            # Position-based selection for better UX
            st.write("**Goalkeepers (select 2):**")
            gk_cols = st.columns(2)
            for i in range(2):
                with gk_cols[i]:
                    gk_options = [''] + \
                        sorted(players_df[players_df['position']
                               == 'Goalkeeper']['web_name'].unique())
                    player = st.selectbox(f"GK {i+1}", gk_options, key=f"gk_{i}")
                    if player:
                        current_squad.append(player)
            st.write("**Defenders (select 5):**")
            def_cols = st.columns(5)
            for i in range(5):
                with def_cols[i]:
                    def_options = [''] + \
                        sorted(players_df[players_df['position']
                               == 'Defender']['web_name'].unique())
                    player = st.selectbox(
                        f"DEF {i+1}", def_options, key=f"def_{i}")
                    if player:
                        current_squad.append(player)
            st.write("**Midfielders (select 5):**")
            mid_cols = st.columns(5)
            for i in range(5):
                with mid_cols[i]:
                    mid_options = [''] + \
                        sorted(players_df[players_df['position']
                               == 'Midfielder']['web_name'].unique())
                    player = st.selectbox(
                        f"MID {i+1}", mid_options, key=f"mid_{i}")
                    if player:
                        current_squad.append(player)
            st.write("**Forwards (select 3):**")
            fwd_cols = st.columns(3)
            for i in range(3):
                with fwd_cols[i]:
                    fwd_options = [''] + \
                        sorted(players_df[players_df['position']
                               == 'Forward']['web_name'].unique())
                    player = st.selectbox(
                        f"FWD {i+1}", fwd_options, key=f"fwd_{i}")
                    if player:
                        current_squad.append(player)
        if len(current_squad) >= 11:
            st.success(f"✅ Selected {len(current_squad)} players")
            if st.button("🔍 Analyze My Squad with AI", type="primary"):
                with st.spinner("Running enhanced squad analysis..."):
                    # Get player data for selected squad
                    squad_data = players_df[players_df['web_name'].isin(
                        current_squad)].copy()
                    if not squad_data.empty:
                        # Get optimal XI from current squad
                        xi_result = predictor.suggest_starting_eleven(
                            squad_data)
                        if "error" not in xi_result:
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.subheader("🎯 Recommended Starting XI")
                                formation = xi_result['formation']
                                captain = xi_result['captain']
                                vice_captain = xi_result['vice_captain']
                                bench_players = xi_result['bench']
                                # Formation and predictions
                                st.info(
                                    f"🏟️ **Best Formation:** {formation['name']} ({xi_result['total_predicted_points']:.1f} predicted points)")
                                # Captain recommendations with reasoning
                                st.success(
                                    f"👑 **Captain:** {captain['web_name']} - {captain['predicted_points']:.1f} pts (Best predicted performance)")
                                st.info(
                                    f"🎖️ **Vice-Captain:** {vice_captain['web_name']} - {vice_captain['predicted_points']:.1f} pts")
                                # Field visualization - prevent reset issue
                                viz_option = st.radio(
                                    "Field View:", ["Interactive", "Simple"], horizontal=True, key="squad_viz")
                                # Store XI result to prevent reset
                                if 'squad_xi_result' not in st.session_state:
                                    st.session_state.squad_xi_result = xi_result
                                stored_result = st.session_state.squad_xi_result
                                if viz_option == "Interactive":
                                    fig = create_alternative_field_viz(
                                        stored_result['players'], formation, captain, vice_captain, bench_players
                                    )
                                    st.plotly_chart(
                                        fig, use_container_width=True)
                                    # Bench for interactive mode
                                    st.markdown("### 🪑 **Bench Players**")
                                    if bench_players:
                                        bench_cols = st.columns(4)
                                        for i in range(min(4, len(bench_players))):
                                            player = bench_players[i]
                                            with bench_cols[i]:
                                                st.markdown(f"""
                                                <div style="
                                                    background: linear-gradient(135deg, #f8f9fa, #e9ecef);
                                                    border: 3px solid #495057;
                                                    border-radius: 15px;
                                                    padding: 15px 10px;
                                                    text-align: center;
                                                    margin: 5px;
                                                    box-shadow: 0 3px 6px rgba(0,0,0,0.2);
                                                ">
                                                    <div style="font-size: 16px; font-weight: bold; color: #000000;">
                                                        {i+1}
                                                    </div>
                                                    <div style="font-weight: bold; color: #000000; margin: 5px 0; font-size: 12px;">
                                                        {player.get('web_name', 'Unknown')}
                                                    </div>
                                                    <div style="color: #495057; font-size: 10px; font-weight: bold;">
                                                        {player.get('position', 'Unknown')}
                                                    </div>
                                                    <div style="color: #28a745; font-weight: bold; font-size: 10px;">
                                                        £{safe_float(player.get('value', 0)):.1f}M
                                                    </div>
                                                    <div style="color: #007bff; font-weight: bold; font-size: 10px;">
                                                        {safe_float(player.get('predicted_points', 0)):.1f} pts
                                                    </div>
                                                </div>
                                                """, unsafe_allow_html=True)
                                    else:
                                        st.info("No bench players available")
                                else:
                                    create_field_visualization(
                                        stored_result['players'], formation, captain, vice_captain, bench_players
                                    )
                                # Transfer suggestions
                                st.subheader("🔄 AI Transfer Suggestions")
                                weak_players = squad_data.nsmallest(
                                    3, 'predicted_points')
                                strong_alternatives = players_df[
                                    ~players_df['web_name'].isin(current_squad)
                                ].nlargest(5, 'value_efficiency')
                                st.write("**Consider Transferring Out:**")
                                for _, player in weak_players.iterrows():
                                    st.write(
                                        f"• {player['web_name']} ({player['position']}) - {player['predicted_points']:.1f} pts, Low efficiency")
                                st.write("**Strong Alternative Options:**")
                                for _, player in strong_alternatives.head(3).iterrows():
                                    st.write(
                                        f"• {player['web_name']} ({player['position']}) - £{player['value']:.1f}M, {player['value_efficiency']:.2f} efficiency")
                            with col2:
                                st.subheader("📊 Squad Statistics")
                                total_value = squad_data['value'].sum()
                                avg_predicted = squad_data['predicted_points'].mean()
                                reliable_starters = sum(
                                    1 for _, p in squad_data.iterrows() if p['starting_probability'] > 0.8)
                                st.metric(
                                    "Squad Value", f"£{total_value:.1f}M")
                                st.metric(
                                    "Avg Predicted Points", f"{avg_predicted:.1f}")
                                st.metric(
                                    "Reliable Starters", f"{reliable_starters}/15")
                                st.metric(
                                    "XI Total Predicted", f"{xi_result['total_predicted_points']:.1f}")
                                # Squad strengths/weaknesses
                                st.subheader("🔍 Squad Analysis")
                                # Position strength analysis
                                pos_strength = {}
                                for pos in ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']:
                                    pos_players = squad_data[squad_data['position'] == pos]
                                    if len(pos_players) > 0:
                                        avg_pred = pos_players['predicted_points'].mean()
                                        pos_strength[pos] = avg_pred
                                if pos_strength:
                                    strongest_pos = max(
                                        pos_strength.items(), key=lambda x: x[1])
                                    weakest_pos = min(
                                        pos_strength.items(), key=lambda x: x[1])
                                    st.success(
                                        f"💪 **Strongest:** {strongest_pos[0]} ({strongest_pos[1]:.1f} avg)")
                                    st.warning(
                                        f"⚠️ **Weakest:** {weakest_pos[0]} ({weakest_pos[1]:.1f} avg)")
                                # Risk analysis
                                rotation_risk = sum(
                                    1 for _, p in squad_data.iterrows() if p['starting_probability'] < 0.7)
                                if rotation_risk > 3:
                                    st.error(
                                        f"🚨 **High Risk:** {rotation_risk} rotation risks")
                                else:
                                    st.success(
                                        f"✅ **Low Risk:** {rotation_risk} rotation risks")
                        else:
                            st.error(
                                "❌ Could not analyze your squad. Please check player selections.")
                    else:
                        st.error("❌ No valid players found in your selection")
        else:
            st.warning(
                f"⚠️ Please select at least 11 players (currently: {len(current_squad)})")
    elif page == "FPL Strategy Chat":
        st.header("💬 Enhanced FPL Strategy Chat")
        # Initialize session state for chat (fixed infinite loop issue)
        if 'chat_messages' not in st.session_state:
            st.session_state.chat_messages = [
                {"role": "assistant", "content": "👋 Hello! I'm your enhanced FPL AI Assistant. I can help with team selection, transfers, captaincy, formations, and strategy. What would you like to know?"}
            ]
        if 'chat_input_counter' not in st.session_state:
            st.session_state.chat_input_counter = 0
        col1, col2 = st.columns([2, 1])
        with col1:
            # Display chat messages
            st.markdown('<div class="chat-container">',
                        unsafe_allow_html=True)
            for message in st.session_state.chat_messages:
                if message["role"] == "user":
                    st.markdown(
                        f'<div class="user-message"><strong>You:</strong> {message["content"]}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="ai-message"><strong>🤖 FPL AI:</strong><br>{message["content"]}</div>',
                        unsafe_allow_html=True
                    )
            st.markdown('</div>', unsafe_allow_html=True)
            # User input with unique key to prevent infinite loop
            user_input = st.text_input(
                "Ask your FPL question:",
                key=f"chat_input_{st.session_state.chat_input_counter}",
                placeholder="e.g., Who should I captain this week?"
            )
            col_send, col_clear = st.columns([1, 1])
            with col_send:
                if st.button("Send 📤", type="primary", key="send_message") and user_input.strip():
                    with st.spinner("🤖 AI is analyzing..."):
                        # Get AI response
                        context = {
                            'total_players': len(players_df),
                            'top_scorers': players_df.nlargest(3, 'total_points')['web_name'].tolist(),
                            'current_gw': 25  # You can make this dynamic
                        }
                        ai_response = chatbot.get_ai_response(
                            user_input, context)
                        # Add to chat history
                        st.session_state.chat_messages.append(
                            {"role": "user", "content": user_input})
                        st.session_state.chat_messages.append(
                            {"role": "assistant", "content": ai_response})
                        # Add to chatbot's internal history
                        chatbot.add_to_history(user_input, ai_response)
                        # Update counter to reset input field
                        st.session_state.chat_input_counter += 1
                        # Rerun to show new messages
                        st.rerun()
            with col_clear:
                if st.button("Clear Chat 🗑️", key="clear_chat"):
                    st.session_state.chat_messages = [
                        {"role": "assistant", "content": "👋 Chat cleared! How can I help you with your FPL strategy?"}
                    ]
                    chatbot.chat_history = []
                    st.session_state.chat_input_counter += 1
                    st.rerun()
        with col2:
            st.subheader("💡 Quick Questions")
            # Quick question buttons that work properly
            quick_questions = [
                "Who should I captain this week?",
                "When should I use my wildcard?",
                "What's the best formation for my team?",
                "How should I plan my transfers?",
                "Which players are good differentials?",
                "What's the current chip strategy?",
                "Who are the best value picks?",
                "How do I improve my rank?"
            ]
            for i, question in enumerate(quick_questions):
                if st.button(f"❓ {question}", key=f"quick_q_{i}"):
                    with st.spinner("🤖 Generating response..."):
                        context = {
                            'quick_question': True,
                            'top_performers': players_df.nlargest(5, 'predicted_points')['web_name'].tolist()
                        }
                        ai_response = chatbot.get_ai_response(
                            question, context)
                        # Add to chat
                        st.session_state.chat_messages.append(
                            {"role": "user", "content": question})
                        st.session_state.chat_messages.append(
                            {"role": "assistant", "content": ai_response})
                        # Add to chatbot's internal history
                        chatbot.add_to_history(question, ai_response)
                        st.rerun()
            st.subheader("⚙️ Chat Info")
            st.info(f"""
            **Enhanced Features:**
            • {'🟢 AI API Active' if api_key else '🟡 Knowledge Base Active'}
            • Context-aware responses
            • FPL-specific expertise
            • Historical analysis integration
            • Current season data
            **Total Messages:** {len(st.session_state.chat_messages)}
            """)
    elif page == "Player Comparison":
        st.header("⚖️ Enhanced Player Comparison")
        col1, col2 = st.columns(2)
        with col1:
            player1 = st.selectbox(
                "Select Player 1:", sorted(players_df['web_name'].unique()), key="player1_select")
        with col2:
            player2 = st.selectbox(
                "Select Player 2:", sorted(players_df['web_name'].unique()), key="player2_select")
        if player1 and player2 and player1 != player2:
            p1_data = players_df[players_df['web_name']
                                 == player1].iloc[0]
            p2_data = players_df[players_df['web_name']
                                 == player2].iloc[0]
            st.subheader("📊 Enhanced Comparison Analysis")
            # Create comprehensive comparison
            comparison_metrics = {
                'Metric': [
                    'Current Points', 'Price (£M)', 'Predicted Next GW', 'Value Efficiency',
                    'Form Score', 'Starting Probability', 'Minutes Played', 'Goals + Assists',
                    'Historical Weight', 'Fixture Adjustment'
                ],
                player1: [
                    safe_float(p1_data.get('total_points', 0)),
                    safe_float(p1_data.get('value', 0)),
                    safe_float(p1_data.get('predicted_points', 0)),
                    safe_float(p1_data.get('value_efficiency', 0)),
                    safe_float(p1_data.get('form_trend', 0)),
                    safe_float(p1_data.get('starting_probability', 0)),
                    safe_float(p1_data.get('minutes', 0)),
                    safe_float(p1_data.get('goals_scored', 0)) +
                    safe_float(p1_data.get('assists', 0)),
                    safe_float(p1_data.get('historical_weight', 0)),
                    safe_float(p1_data.get('fixture_adjustment', 0))
                ],
                player2: [
                    safe_float(p2_data.get('total_points', 0)),
                    safe_float(p2_data.get('value', 0)),
                    safe_float(p2_data.get('predicted_points', 0)),
                    safe_float(p2_data.get('value_efficiency', 0)),
                    safe_float(p2_data.get('form_trend', 0)),
                    safe_float(p2_data.get('starting_probability', 0)),
                    safe_float(p2_data.get('minutes', 0)),
                    safe_float(p2_data.get('goals_scored', 0)) +
                    safe_float(p2_data.get('assists', 0)),
                    safe_float(p2_data.get('historical_weight', 0)),
                    safe_float(p2_data.get('fixture_adjustment', 0))
                ]
            }
            comparison_df = pd.DataFrame(comparison_metrics)
            st.dataframe(comparison_df, use_container_width=True)
            # AI-powered comparison analysis
            col_rec, col_viz = st.columns([1, 1])
            with col_rec:
                st.subheader("🤖 AI Recommendation")
                # Create comparison context for AI
                comparison_context = f"""
                Player 1: {player1} - {p1_data['position']} - £{p1_data['value']:.1f}M
                Current Points: {p1_data.get('total_points', 0)}
                Predicted Next GW: {p1_data.get('predicted_points', 0):.1f}
                Value Efficiency: {p1_data.get('value_efficiency', 0):.2f}
                Player 2: {player2} - {p2_data['position']} - £{p2_data['value']:.1f}M  
                Current Points: {p2_data.get('total_points', 0)}
                Predicted Next GW: {p2_data.get('predicted_points', 0):.1f}
                Value Efficiency: {p2_data.get('value_efficiency', 0):.2f}
                """
                if st.button("🔍 Get AI Analysis", type="primary"):
                    with st.spinner("🤖 AI analyzing players..."):
                        ai_analysis = chatbot.get_ai_response(
                            f"Compare these two FPL players and recommend which is better: {comparison_context}",
                            {"comparison": True}
                        )
                        st.markdown(ai_analysis)
                # Quick comparison insights
                st.subheader("⚡ Quick Insights")
                # Better predicted points
                if p1_data.get('predicted_points', 0) > p2_data.get('predicted_points', 0):
                    st.success(
                        f"🎯 **Next GW:** {player1} predicted higher ({p1_data.get('predicted_points', 0):.1f} vs {p2_data.get('predicted_points', 0):.1f})")
                else:
                    st.success(
                        f"🎯 **Next GW:** {player2} predicted higher ({p2_data.get('predicted_points', 0):.1f} vs {p1_data.get('predicted_points', 0):.1f})")
                # Better value
                if p1_data.get('value_efficiency', 0) > p2_data.get('value_efficiency', 0):
                    st.info(
                        f"💰 **Value:** {player1} better efficiency ({p1_data.get('value_efficiency', 0):.2f})")
                else:
                    st.info(
                        f"💰 **Value:** {player2} better efficiency ({p2_data.get('value_efficiency', 0):.2f})")
                # Starting probability
                if p1_data.get('starting_probability', 0) > p2_data.get('starting_probability', 0):
                    st.info(
                        f"⚡ **Reliability:** {player1} more likely to start ({p1_data.get('starting_probability', 0):.0%})")
                else:
                    st.info(
                        f"⚡ **Reliability:** {player2} more likely to start ({p2_data.get('starting_probability', 0):.0%})")
            with col_viz:
                st.subheader("📊 Visual Comparison")
                # Radar chart comparison
                metrics_for_radar = ['predicted_points',
                                     'value_efficiency', 'form_trend', 'starting_probability']
                fig = go.Figure()
                # Normalize values for radar chart
                p1_values = []
                p2_values = []
                labels = []
                for metric in metrics_for_radar:
                    max_val = max(p1_data.get(metric, 0),
                                  p2_data.get(metric, 0))
                    if max_val > 0:
                        p1_norm = p1_data.get(metric, 0) / max_val
                        p2_norm = p2_data.get(metric, 0) / max_val
                    else:
                        p1_norm = p2_norm = 0
                    p1_values.append(p1_norm)
                    p2_values.append(p2_norm)
                    labels.append(metric.replace('_', ' ').title())
                fig.add_trace(go.Scatterpolar(
                    r=p1_values,
                    theta=labels,
                    fill='toself',
                    name=player1,
                    line_color='#37003c'
                ))
                fig.add_trace(go.Scatterpolar(
                    r=p2_values,
                    theta=labels,
                    fill='toself',
                    name=player2,
                    line_color='#00ff87'
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True,
                    title="Player Comparison Radar"
                )
                st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()