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

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #37003c;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .player-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: #f8f9fa;
    }
    .starting-xi-card {
        border: 2px solid #37003c;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: linear-gradient(135deg, #37003c 0%, #563d7c 100%);
        color: white;
    }
    .bench-card {
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.3rem 0;
        background: #fff3cd;
    }
    .chat-container {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        background: #f8f9fa;
        max-height: 500px;
        overflow-y: auto;
    }
    .chat-message {
        margin: 0.5rem 0;
        padding: 0.8rem;
        border-radius: 8px;
    }
    .user-message {
        background: #e3f2fd;
        margin-left: 2rem;
        text-align: right;
    }
    .ai-message {
        background: #f3e5f5;
        margin-right: 2rem;
    }
    .formation-display {
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        color: #37003c;
        margin: 1rem 0;
        padding: 0.5rem;
        background: #f0f0f0;
        border-radius: 8px;
    }
    .stButton > button {
        background: #37003c;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 2rem;
    }
</style>
""", unsafe_allow_html=True)

class FPLDataManager:
    """Enhanced FPL data manager with historical performance tracking"""
    
    def __init__(self):
        self.base_url = "https://fantasy.premierleague.com/api/"
        self.current_season_data = None
        self.players_data = None
        self.teams_data = None
        self.historical_performance = {}
        
    @st.cache_data(ttl=3600)
    def fetch_current_season_data(_self):
        """Fetch current season data from FPL API"""
        try:
            response = requests.get(f"{_self.base_url}bootstrap-static/")
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            st.error(f"Error fetching FPL data: {str(e)}")
            return None
    
    @st.cache_data(ttl=7200)
    def fetch_player_historical_data(_self, player_id: int):
        """Fetch historical performance data for a player"""
        try:
            response = requests.get(f"{_self.base_url}element-summary/{player_id}/")
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None
    
    def get_players_dataframe(self):
        """Enhanced player data with historical performance metrics"""
        if self.current_season_data is None:
            self.current_season_data = self.fetch_current_season_data()
        
        if self.current_season_data:
            players_df = pd.DataFrame(self.current_season_data['elements'])
            teams_df = pd.DataFrame(self.current_season_data['element_types'])
            
            # Add position names
            position_map = {pos['id']: pos['singular_name'] for pos in self.current_season_data['element_types']}
            players_df['position'] = players_df['element_type'].map(position_map)
            
            # Add team names
            team_map = {team['id']: team['name'] for team in self.current_season_data['teams']}
            players_df['team_name'] = players_df['team'].map(team_map)
            
            # Calculate enhanced metrics
            players_df['value'] = players_df['now_cost'] / 10
            players_df['points_per_million'] = players_df['total_points'] / players_df['value']
            players_df['form_float'] = pd.to_numeric(players_df['form'], errors='coerce')
            
            # Add historical performance score
            players_df['historical_score'] = self._calculate_historical_score(players_df)
            
            # Calculate consistency score
            players_df['consistency_score'] = self._calculate_consistency_score(players_df)
            
            # Enhanced starting XI probability
            players_df['starting_xi_probability'] = self._calculate_starting_xi_probability(players_df)
            
            # Ensure numeric columns
            numeric_cols = [
                'total_points', 'points_per_game', 'minutes', 'influence', 
                'creativity', 'threat', 'now_cost', 'historical_score',
                'consistency_score', 'starting_xi_probability'
            ]
            
            for col in numeric_cols:
                if col in players_df.columns:
                    players_df[col] = pd.to_numeric(players_df[col], errors='coerce').fillna(0)
                    
            return players_df
        
        return pd.DataFrame()
    
    def _calculate_historical_score(self, df):
        """Calculate historical performance score"""
        # Simulate historical data based on current performance
        # In a real implementation, this would use actual historical API data
        historical_scores = []
        
        for _, player in df.iterrows():
            # Base score on current season performance
            base_score = player.get('total_points', 0)
            
            # Add consistency factor based on minutes played
            minutes = player.get('minutes', 0)
            consistency_bonus = min(minutes / 2000, 1.5)  # Up to 1.5x bonus for consistent players
            
            # Position-specific adjustments
            position = player.get('element_type', 1)
            position_multiplier = {1: 0.8, 2: 0.9, 3: 1.1, 4: 1.2}.get(position, 1.0)
            
            final_score = base_score * consistency_bonus * position_multiplier
            historical_scores.append(final_score)
        
        return historical_scores
    
    def _calculate_consistency_score(self, df):
        """Calculate player consistency score"""
        consistency_scores = []
        
        for _, player in df.iterrows():
            points = player.get('total_points', 0)
            minutes = player.get('minutes', 0)
            form = player.get('form', '0')
            
            try:
                form_float = float(form)
            except:
                form_float = 0
            
            # Players with good minutes and stable form get higher scores
            if minutes > 1500:  # Regular starter
                consistency = min(form_float * (minutes / 2000), 10)
            else:
                consistency = form_float * 0.5  # Penalize rotation risk
                
            consistency_scores.append(consistency)
        
        return consistency_scores
    
    def _calculate_starting_xi_probability(self, df):
        """Calculate probability of being in starting XI"""
        probabilities = []
        
        for _, player in df.iterrows():
            base_prob = 0.5
            
            # Minutes played factor
            minutes = player.get('minutes', 0)
            if minutes > 2000:
                minutes_factor = 0.9
            elif minutes > 1500:
                minutes_factor = 0.7
            elif minutes > 1000:
                minutes_factor = 0.5
            else:
                minutes_factor = 0.2
            
            # Form factor
            form = pd.to_numeric(player.get('form', 0), errors='coerce')
            form_factor = min(form / 10, 0.3)  # Up to 0.3 bonus for good form
            
            # Points per game factor
            ppg = player.get('points_per_game', 0)
            ppg_factor = min(float(ppg) / 20, 0.2)  # Up to 0.2 bonus for high PPG
            
            final_prob = min(base_prob + minutes_factor + form_factor + ppg_factor, 1.0)
            probabilities.append(final_prob)
        
        return probabilities

class EnhancedFPLPredictor:
    def __init__(self, players_df):
        if isinstance(players_df, list):
            players_df = pd.DataFrame(players_df)
        self.players_df = players_df.copy()
        self.position_limits = {"Goalkeeper": 2, "Defender": 5, "Midfielder": 5, "Forward": 3}
        self.team_limit = 3
        self._initialize_required_columns()

    def _initialize_required_columns(self):
        df = self.players_df
        if "minutes" not in df.columns:
            df["minutes"] = 0
        if "games_played" not in df.columns:
            self.calculate_games_played()
        if "total_points_per_game" not in df.columns:
            self.calculate_total_points_per_game()
        if "value" not in df.columns:
            self.calculate_value()
        if "form_float" not in df.columns:
            self.calculate_form_float()
        if "historical_score" not in df.columns:
            self.calculate_historical_score()
        if "consistency_score" not in df.columns:
            self.calculate_consistency_score()
        if "starting_xi_probability" not in df.columns:
            self.calculate_starting_xi_probability()
        self.players_df = df

    def calculate_games_played(self):
        self.players_df["games_played"] = (self.players_df["minutes"] / 90).fillna(0).replace([float("inf"), -float("inf")], 0)
    def calculate_total_points_per_game(self):
        if "games_played" not in self.players_df.columns:
            self.calculate_games_played()
        self.players_df["total_points_per_game"] = (
            self.players_df["total_points"] / self.players_df["games_played"]
        ).replace([float("inf"), -float("inf")], 0).fillna(0)
    def calculate_value(self):
        self.players_df["value"] = (self.players_df["now_cost"] / 10).fillna(0)
    def calculate_form_float(self):
        self.players_df["form_float"] = pd.to_numeric(self.players_df["form"], errors="coerce").fillna(0)
    def calculate_historical_score(self):
        self.players_df["historical_score"] = (self.players_df["total_points"] * 0.8).fillna(0)
    def calculate_consistency_score(self):
        self.players_df["consistency_score"] = self.players_df["total_points"].rolling(window=5, min_periods=1).mean().fillna(self.players_df["total_points"].mean()).fillna(0)
    def calculate_starting_xi_probability(self):
        prob = 0.5
        self.players_df["starting_xi_probability"] = prob
        self.players_df["starting_xi_probability"] += self.players_df["minutes"].apply(
            lambda x: 0.3 if x > 2000 else (0.2 if x > 1500 else (0.1 if x > 1000 else 0))
        )
        self.players_df["starting_xi_probability"] += self.players_df["form_float"] / 10
        self.players_df["starting_xi_probability"] += self.players_df["total_points_per_game"] / 20
        self.players_df["starting_xi_probability"] = self.players_df["starting_xi_probability"].clip(0, 1)

    def optimize_team_selection(self, budget: float) -> dict:
        self._initialize_required_columns()
        filtered_players = self.players_df[self.players_df["now_cost"] <= budget].copy()
        selected_players = []
        remaining_budget = budget
        for position in ["Goalkeeper", "Defender", "Midfielder", "Forward"]:
            pos_limit = self.position_limits[position]
            candidates = filtered_players[filtered_players["position"].astype(str).str.strip().str.title() == position]
            candidates = candidates.sort_values(by="total_points_per_game", ascending=False)
            pos_selected = []
            for _, player in candidates.iterrows():
                if len(pos_selected) < pos_limit and player["now_cost"] <= remaining_budget:
                    pos_selected.append(player)
                    remaining_budget -= player["now_cost"]
            selected_players.extend(pos_selected)
        team_counts = {}
        final_team = []
        for player in selected_players:
            team = player["team"] if "team" in player else None
            if team is None:
                continue
            if team_counts.get(team, 0) < self.team_limit:
                final_team.append(player)
                team_counts[team] = team_counts.get(team, 0) + 1
        total_cost = sum(player["now_cost"] if "now_cost" in player else 0 for player in final_team)
        return {
            "team": final_team,
            "total_cost": total_cost,
        }

    def suggest_optimal_starting_eleven(self, team_players=None):
        if team_players is None:
            team_players = self.players_df
        if isinstance(team_players, list):
            team_players = pd.DataFrame(team_players)
        if not isinstance(team_players, pd.DataFrame):
            raise ValueError("team_players must be a DataFrame or list of records.")
        if "position" not in team_players.columns or "total_points_per_game" not in team_players.columns:
            raise ValueError("team_players must have 'position' and 'total_points_per_game' columns.")
        team_players = team_players.copy()
        team_players["position"] = team_players["position"].astype(str).str.strip().str.title()
        formations = [
            {"Defender": 3, "Midfielder": 4, "Forward": 3},
            {"Defender": 3, "Midfielder": 5, "Forward": 2},
            {"Defender": 4, "Midfielder": 4, "Forward": 2},
        ]
        for formation in formations:
            xi = []
            gks = team_players[team_players["position"] == "Goalkeeper"]
            if len(gks) < 1:
                continue
            gk = gks.sort_values("total_points_per_game", ascending=False).iloc[0]
            xi.append(gk.to_dict())
            valid = True
            for pos in ["Defender", "Midfielder", "Forward"]:
                players = team_players[team_players["position"] == pos]
                if len(players) < formation[pos]:
                    valid = False
                    break
                pos_players = players.sort_values("total_points_per_game", ascending=False).iloc[: formation[pos]]
                xi.extend(pos_players.to_dict(orient="records"))
            if valid and len(xi) == 11:
                return {
                    "players": pd.DataFrame(xi),
                    "formation": formation,
                }
        return {"error": "Insufficient players for any valid formation"}

class FPLChatBot:
    """Enhanced chatbot using EuriAI for FPL strategy discussions"""
    
    def __init__(self, euriai_api_key: str = None):
        self.euriai_api_key = euriai_api_key
        self.chat_history = []
        
        # FPL knowledge base
        self.fpl_knowledge = {
            "captain_strategy": "Choose captains based on fixtures, form, and expected minutes. Premium players from top teams are typically safest.",
            "wildcard_timing": "Use first wildcard GW7-10 after initial team assessment. Second wildcard before double gameweeks (GW19-25).",
            "transfer_strategy": "Don't rush early transfers. Bank transfers for double gameweeks. Consider price changes.",
            "chip_usage": "Triple Captain on premium players during double gameweeks. Bench Boost when bench has good fixtures.",
            "formation_advice": "3-5-2 great for premium midfielders. 3-4-3 for attacking approach. 4-4-2 most balanced.",
            "budget_allocation": "Spend 60-65% on outfield players. Don't overspend on goalkeeper. Invest in consistent starters."
        }
    
    def get_contextual_response(self, user_message: str, context_data: Dict = None) -> str:
        """Generate contextual response using EuriAI or fallback"""
        
        # Add to chat history
        self.chat_history.append({"role": "user", "content": user_message})
        
        if self.euriai_api_key:
            try:
                # Use EuriAI for enhanced responses
                response = self._get_euriai_response(user_message, context_data)
                self.chat_history.append({"role": "assistant", "content": response})
                return response
            except Exception as e:
                st.error(f"EuriAI API error: {e}")
                return self._get_fallback_response(user_message, context_data)
        else:
            return self._get_fallback_response(user_message, context_data)
    
    def _get_euriai_response(self, message: str, context_data: Dict = None) -> str:
        """Get response from EuriAI API"""
        
        # Build context
        system_prompt = """You are an expert FPL (Fantasy Premier League) analyst with deep knowledge of:
        - Player performance analysis and predictions
        - Transfer strategies and timing
        - Captaincy decisions and chip usage
        - Formation tactics and starting XI selection
        - Budget management and value picks
        - Historical trends and statistical analysis
        
        Provide specific, actionable advice that helps users improve their FPL performance. 
        Use data-driven insights and consider current season context."""
        
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add chat history for context
        messages.extend(self.chat_history[-6:])  # Last 6 messages for context
        
        if context_data:
            context_message = f"Current context: {json.dumps(context_data, indent=2)}"
            messages.append({"role": "system", "content": context_message})
        
        messages.append({"role": "user", "content": message})
        
        # Make API call to EuriAI
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.euriai_api_key}"
        }
        
        payload = {
            "model": "gemini-2.5-pro",
            "max_tokens": 800,
            "messages": messages,
            "temperature": 0.5
        }
        
        response = requests.post(
            "https://api.euron.one/api/v1/euri/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            data = response.json()
            return data['content'][0]['text']
        else:
            raise Exception(f"API error: {response.status_code}")
    
    def _get_fallback_response(self, message: str, context_data: Dict = None) -> str:
        """Fallback response system without API"""
        
        message_lower = message.lower()
        
        # Pattern matching for common questions
        if any(word in message_lower for word in ['captain', 'c', 'armband']):
            return self._get_captain_advice(context_data)
        elif any(word in message_lower for word in ['transfer', 'buy', 'sell']):
            return self._get_transfer_advice(context_data)
        elif any(word in message_lower for word in ['formation', 'starting', 'xi', '11']):
            return self._get_formation_advice(context_data)
        elif any(word in message_lower for word in ['chip', 'wildcard', 'bench boost', 'triple']):
            return self._get_chip_advice(context_data)
        elif any(word in message_lower for word in ['budget', 'money', 'price', 'value']):
            return self._get_budget_advice(context_data)
        else:
            return self._get_general_advice(message, context_data)
    
    def _get_captain_advice(self, context_data):
        """Generate captain advice"""
        advice = "🎯 **Captain Selection Strategy:**\n\n"
        advice += "• Choose players with favorable fixtures (FDR ≤ 3)\n"
        advice += "• Prioritize consistent performers over explosive differentials\n"
        advice += "• Consider double gameweek players when available\n"
        advice += "• Top options usually: Haaland, Salah, premium FWDs/MIDs\n"
        advice += "• Avoid captaining defenders unless exceptional fixtures\n"
        
        if context_data:
            if 'gameweek' in context_data:
                advice += f"\n• For GW{context_data['gameweek']}: Check fixture difficulty and team news"
        
        return advice
    
    def _get_transfer_advice(self, context_data):
        """Generate transfer advice"""
        advice = "🔄 **Transfer Strategy:**\n\n"
        advice += "• Don't rush early transfers - assess team performance first\n"
        advice += "• Bank transfers for double gameweeks when possible\n"
        advice += "• Consider price changes when planning ahead\n"
        advice += "• Target players with good fixture runs (3-4 GWs)\n"
        advice += "• Avoid sideways transfers unless for fixture swings\n"
        advice += "• Monitor team news before deadline\n"
        
        return advice
    
    def _get_formation_advice(self, context_data):
        """Generate formation advice"""
        advice = "⚡ **Formation Selection:**\n\n"
        advice += "• **3-5-2**: Best for premium midfielder strategy\n"
        advice += "• **3-4-3**: Aggressive, good for chasing points\n"
        advice += "• **4-4-2**: Most balanced, safe option\n"
        advice += "• **4-3-3**: Good mix of attack and stability\n"
        advice += "• **5-3-2**: When you have premium defenders\n\n"
        advice += "Choose based on your premium players' positions!"
        
        return advice
    
    def _get_chip_advice(self, context_data):
        """Generate chip usage advice"""
        advice = "💎 **Chip Usage Strategy:**\n\n"
        advice += "• **Wildcard 1**: GW7-10 (international break)\n"
        advice += "• **Wildcard 2**: GW19-25 (prepare for DGWs)\n"
        advice += "• **Triple Captain**: Premium player in DGW\n"
        advice += "• **Bench Boost**: When bench has good fixtures\n"
        advice += "• **Free Hit**: Blank gameweeks or unique opportunities\n"
        
        return advice
    
    def _get_budget_advice(self, context_data):
        """Generate budget management advice"""
        advice = "💰 **Budget Management:**\n\n"
        advice += "• Spend 60-65% on outfield players\n"
        advice += "• Don't overspend on goalkeepers (4.5M max)\n"
        advice += "• Invest in 2-3 premium players (8M+)\n"
        advice += "• Balance with 4.5M enablers in defense\n"
        advice += "• Monitor price changes for team value growth\n"
        
        return advice
    
    def _get_general_advice(self, message, context_data):
        """General FPL advice"""
        advice = "🏆 **General FPL Tips:**\n\n"
        advice += "• Research fixtures and underlying stats\n"
        advice += "• Follow team news and press conferences\n"
        advice += "• Consider differential picks for rank improvement\n"
        advice += "• Stay patient - FPL is a season-long game\n"
        advice += "• Join communities for insights and discussions\n"
        
        if context_data and 'current_rank' in context_data:
            rank = context_data['current_rank']
            if rank > 1000000:
                advice += "\n• Focus on template picks to climb ranks safely"
            else:
                advice += "\n• Consider differentials to push for higher ranks"
        
        return advice

def main():
    """Enhanced main application with integrated chat"""
    
    st.markdown('<h1 class="main-header">⚽ FPL AI Assistant with Strategy Chat</h1>', unsafe_allow_html=True)
    
    # Initialize data manager and chatbot
    data_manager = FPLDataManager()
    
    st.sidebar.title("🔧 Settings")
    
    # EuriAI API Key input
    euriai_key = st.sidebar.text_input(
        "EuriAI API Key (Optional)", 
        type="password",
        help="Enter your EuriAI API key for enhanced AI chat responses"
    )
    
    # Initialize chatbot
    if euriai_key:
        os.environ["EURIAI_API_KEY"] = euriai_key
        chatbot = FPLChatBot(euriai_key)
        st.sidebar.success("✅ EuriAI API key set - Enhanced chat enabled!")
    else:
        chatbot = FPLChatBot()
        st.sidebar.info("💡 Add EuriAI API key for enhanced AI chat responses")
    
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "Team Optimization", 
        "Squad Analysis", 
        "FPL Strategy Chat",
        "Chip Strategy", 
        "Player Comparison"
    ])
    
    # Load player data
    with st.spinner("Loading FPL data..."):
        players_df = data_manager.get_players_dataframe()
    
    if players_df.empty:
        st.error("Unable to load FPL data. Please check your internet connection and try again.")
        return
    
    predictor = EnhancedFPLPredictor(players_df)
    
    if page == "Team Optimization":
        st.header("🎯 Optimal Team Selection & Starting XI")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            budget = st.slider("Budget (£M)", 95.0, 105.0, 100.0, 0.5)
            
            if st.button("Generate Optimal Team with Starting XI", type="primary"):
                with st.spinner("Optimizing team selection and starting XI..."):
                    # Generate optimal 15-man squad
                    result = predictor.optimize_team_selection(budget)
                    
                    if 'team' in result:
                        st.success(f"✅ Optimal team generated! Cost: £{result['total_cost']:.1f}M")
                        
                        # Get starting XI suggestion
                        xi_result = predictor.suggest_optimal_starting_eleven(result['team'])
                        
                        if 'players' in xi_result:
                            # Display Starting XI
                            st.subheader("🔥 Recommended Starting XI")
                            formation_name = xi_result['formation_name']
                            st.markdown(f'<div class="formation-display">Formation: {formation_name}</div>', unsafe_allow_html=True)
                            
                            # Captain suggestions
                            captain = xi_result['captain_suggestion']
                            vice_captain = xi_result['vice_captain_suggestion']
                            
                            col_cap, col_vice = st.columns(2)
                            with col_cap:
                                st.info(f"👑 **Captain**: {captain['name']} ({captain['predicted_points']:.1f} pts)")
                            with col_vice:
                                st.info(f"🎖️ **Vice-Captain**: {vice_captain['name']} ({vice_captain['predicted_points']:.1f} pts)")
                            
                            # Display XI by position
                            xi_by_position = {}
                            for player in xi_result['players']:
                                pos = player['position']
                                if pos not in xi_by_position:
                                    xi_by_position[pos] = []
                                xi_by_position[pos].append(player)
                            
                            for pos in ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']:
                                if pos in xi_by_position:
                                    st.markdown(f"**{pos}s:**")
                                    for player in xi_by_position[pos]:
                                        is_captain = player['name'] == captain['name']
                                        is_vice = player['name'] == vice_captain['name']
                                        captain_badge = " (C)" if is_captain else " (VC)" if is_vice else ""
                                        
                                        st.markdown(
                                            f'<div class="starting-xi-card">'
                                            f'<strong>{player["name"]}{captain_badge}</strong> ({player["team"]})<br>'
                                            f'£{player["cost"]:.1f}M • {player["predicted_points"]:.1f} pts • '
                                            f'Start Prob: {player.get("starting_xi_probability", 0.5):.0%}'
                                            f'</div>', 
                                            unsafe_allow_html=True
                                        )
                            
                            # Display Bench
                            st.subheader("🪑 Bench")
                            bench_order = sorted(xi_result['bench'], 
                                               key=lambda x: x.get('starting_xi_probability', 0) * x.get('predicted_points', 0), 
                                               reverse=True)
                            
                            for i, player in enumerate(bench_order, 1):
                                st.markdown(
                                    f'<div class="bench-card">'
                                    f'<strong>{i}. {player["name"]}</strong> ({player["position"]}, {player["team"]})<br>'
                                    f'£{player["cost"]:.1f}M • {player["predicted_points"]:.1f} pts'
                                    f'</div>', 
                                    unsafe_allow_html=True
                                )
                            
                            # Show alternative formations
                            if xi_result.get('alternative_formations'):
                                with st.expander("🔄 Alternative Formations"):
                                    for alt_name, alt_data in list(xi_result['alternative_formations'].items())[:2]:
                                        st.write(f"**{alt_name}**: {alt_data['description']}")
                                        st.write(f"Predicted Points: {alt_data['total_predicted_points']:.1f}")
                                        st.write("---")
                        
                        else:
                            st.error("Could not generate starting XI from the optimal team")
                        
                        # Display full 15-man squad breakdown
                        with st.expander("📋 Full 15-Man Squad Breakdown"):
                            positions = ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']
                            for pos in positions:
                                pos_players = [p for p in result['team'] if p['position'] == pos]
                                if pos_players:
                                    st.subheader(f"{pos}s ({len(pos_players)})")
                                    for player in pos_players:
                                        col_a, col_b, col_c, col_d = st.columns([3, 1, 1, 1])
                                        with col_a:
                                            st.write(f"**{player['web_name']}** ({player['team']})")
                                        with col_b:
                                            st.write(f"£{player['cost']:.1f}M")
                                        with col_c:
                                            st.write(f"{player['predicted_points']:.1f} pts")
                                        with col_d:
                                            st.write(f"Form: {player.get('form', 0):.1f}")
        
        with col2:
            st.markdown("### 📊 Team Optimization")
            st.info("""
            **Enhanced Features:**
            - ML-powered predictions
            - Historical performance analysis
            - Starting XI probability
            - Multiple formation options
            - Captain/VC suggestions
            - Bench optimization
            
            **Formation Rules:**
            - 2 Goalkeepers, 5 Defenders
            - 5 Midfielders, 3 Forwards
            - Max 3 players per team
            - £100M budget constraint
            """)
            
            euriai_status = "✅ Active" if euriai_key else "⚠️ Not configured"
            st.info(f"**EuriAI Status:** {euriai_status}")
    
    elif page == "Squad Analysis":
        st.header("📋 Enhanced Squad Analysis")
        
        st.info("Enter your current 15 players for AI-powered starting XI optimization and transfer suggestions")
        
        with st.expander("Input Your Current Squad", expanded=True):
            current_team = []
            cols = st.columns(3)
            
            for i in range(15):
                with cols[i % 3]:
                    player = st.selectbox(
                        f"Player {i+1}:",
                        options=[''] + list(players_df['web_name'].unique()),
                        key=f"player_{i}"
                    )
                    if player:
                        current_team.append(player)
        
        if len(current_team) == 15:
            if st.button("🔍 Analyze Squad with Enhanced AI", type="primary"):
                # Get enhanced player details
                team_details = []
                total_squad_value = 0
                
                for player_name in current_team:
                    player_data = players_df[players_df['web_name'] == player_name].iloc[0]
                    enhanced_predicted_points = predictor.calculate_enhanced_predicted_points().loc[player_data.name, 'predicted_points']
                    
                    player_info = {
                        'name': player_data['web_name'],
                        'position': player_data['position'],
                        'team': player_data['team_name'],
                        'cost': player_data['value'],
                        'predicted_points': enhanced_predicted_points,
                        'form': player_data['form_float'],
                        'starting_xi_probability': player_data['starting_xi_probability'],
                        'consistency_score': player_data['consistency_score'],
                        'historical_score': player_data['historical_score']
                    }
                    team_details.append(player_info)
                    total_squad_value += player_data['value']
                
                # Get optimal starting XI
                xi_result = predictor.suggest_optimal_starting_eleven(team_details)
                
                if 'players' in xi_result:
                    col1, col2 = st.columns([3, 2])
                    
                    with col1:
                        st.success("🎯 Optimized Starting XI Generated!")
                        
                        formation = xi_result['formation']
                        formation_name = xi_result['formation_name']
                        st.markdown(f'<div class="formation-display">{formation_name}</div>', unsafe_allow_html=True)
                        
                        # Captain recommendations
                        captain = xi_result['captain_suggestion']
                        vice_captain = xi_result['vice_captain_suggestion']
                        
                        st.info(f"👑 **Recommended Captain**: {captain['name']} ({captain['predicted_points']:.1f} pts)")
                        st.info(f"🎖️ **Vice-Captain**: {vice_captain['name']} ({vice_captain['predicted_points']:.1f} pts)")
                        
                        # Starting XI with enhanced metrics
                        st.subheader("🔥 Your Starting XI")
                        xi_df = pd.DataFrame(xi_result['players'])
                        xi_df = xi_df.sort_values(['position', 'predicted_points'], ascending=[True, False])
                        
                        for pos in ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']:
                            pos_players = xi_df[xi_df['position'] == pos]
                            if not pos_players.empty:
                                st.write(f"**{pos}s:**")
                                for _, player in pos_players.iterrows():
                                    is_captain = player['name'] == captain['name']
                                    is_vice = player['name'] == vice_captain['name']
                                    captain_badge = " (C)" if is_captain else " (VC)" if is_vice else ""
                                    
                                    st.markdown(
                                        f'<div class="starting-xi-card">'
                                        f'<strong>{player["name"]}{captain_badge}</strong> ({player["team"]})<br>'
                                        f'Predicted: {player["predicted_points"]:.1f} pts • '
                                        f'Start Prob: {player["starting_xi_probability"]:.0%} • '
                                        f'Form: {player["form"]:.1f}'
                                        f'</div>',
                                        unsafe_allow_html=True
                                    )
                        
                        # Bench analysis
                        st.subheader("🪑 Bench Analysis")
                        bench_df = pd.DataFrame(xi_result['bench'])
                        bench_df = bench_df.sort_values('starting_xi_probability', ascending=False)
                        
                        for i, (_, player) in enumerate(bench_df.iterrows(), 1):
                            st.markdown(
                                f'<div class="bench-card">'
                                f'<strong>{i}. {player["name"]}</strong> ({player["position"]}, {player["team"]})<br>'
                                f'Predicted: {player["predicted_points"]:.1f} pts • '
                                f'Start Prob: {player["starting_xi_probability"]:.0%}'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                    
                    with col2:
                        st.subheader("📊 Squad Statistics")
                        
                        # Key metrics
                        total_predicted = sum(p['predicted_points'] for p in xi_result['players'])
                        avg_start_prob = np.mean([p['starting_xi_probability'] for p in team_details])
                        
                        st.metric("Total Squad Value", f"£{total_squad_value:.1f}M")
                        st.metric("Starting XI Predicted Points", f"{total_predicted:.1f}")
                        st.metric("Average Starting Probability", f"{avg_start_prob:.1%}")
                        
                        # Position distribution
                        st.subheader("Position Distribution")
                        pos_counts = pd.Series([p['position'] for p in team_details]).value_counts()
                        st.bar_chart(pos_counts)
                        
                        # Alternative formations
                        if xi_result.get('alternative_formations'):
                            st.subheader("🔄 Alternative Formations")
                            alt_formations = list(xi_result['alternative_formations'].items())[:3]
                            
                            for alt_name, alt_data in alt_formations:
                                with st.container():
                                    st.write(f"**{alt_name}**")
                                    st.write(f"Points: {alt_data['total_predicted_points']:.1f}")
                                    st.write(f"Strategy: {alt_data['description']}")
                                    st.write("---")
        
        else:
            st.warning(f"Please select all 15 players. Currently selected: {len(current_team)}")
    
    elif page == "FPL Strategy Chat":
        st.header("💬 FPL Strategy Chat Assistant")
        
        # Initialize session state for chat
        if 'chat_messages' not in st.session_state:
            st.session_state.chat_messages = [
                {"role": "assistant", "content": "👋 Hello! I'm your FPL Strategy Assistant. Ask me anything about team selection, transfers, captaincy, formations, chip usage, or any other FPL strategy questions!"}
            ]
        
        # Chat interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display chat messages
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            
            for message in st.session_state.chat_messages:
                if message["role"] == "user":
                    st.markdown(
                        f'<div class="chat-message user-message">'
                        f'<strong>You:</strong> {message["content"]}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="chat-message ai-message">'
                        f'<strong>🤖 FPL Assistant:</strong><br>{message["content"]}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # User input
            user_input = st.text_input("Ask your FPL question:", key="chat_input", placeholder="e.g., Who should I captain this week?")
            
            col_send, col_clear = st.columns([1, 1])
            with col_send:
                if st.button("Send 📤", type="primary") and user_input:
                    # Get current gameweek context
                    current_gw = st.session_state.get('current_gameweek', 1)
                    context_data = {
                        'gameweek': current_gw,
                        'total_players': len(players_df),
                        'current_date': datetime.now().strftime('%Y-%m-%d')
                    }
                    
                    # Get AI response
                    with st.spinner("🤔 Thinking..."):
                        response = chatbot.get_contextual_response(user_input, context_data)
                    
                    # Add to chat history
                    st.session_state.chat_messages.append({"role": "user", "content": user_input})
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})
                    
                    # Rerun to update chat display
                    st.rerun()
            
            with col_clear:
                if st.button("Clear Chat 🗑️"):
                    st.session_state.chat_messages = [
                        {"role": "assistant", "content": "👋 Hello! I'm your FPL Strategy Assistant. How can I help you today?"}
                    ]
                    st.rerun()
        
        with col2:
            st.subheader("💡 Quick Questions")
            
            # Quick question buttons
            quick_questions = [
                "Who should I captain this week?",
                "When should I use my wildcard?",
                "What's the best formation to use?",
                "How should I plan my transfers?",
                "Which chips should I use when?",
                "How do I improve my team value?",
                "What are good differential picks?",
                "How do I analyze fixtures?"
            ]
            
            for question in quick_questions:
                if st.button(f"❓ {question}", key=f"quick_{question}"):
                    # Simulate user asking this question
                    context_data = {
                        'gameweek': st.session_state.get('current_gameweek', 1),
                        'quick_question': True
                    }
                    
                    with st.spinner("🤔 Generating response..."):
                        response = chatbot.get_contextual_response(question, context_data)
                    
                    # Add to chat
                    st.session_state.chat_messages.append({"role": "user", "content": question})
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})
                    st.rerun()
            
            st.subheader("⚙️ Chat Settings")
            current_gw = st.number_input("Current Gameweek", 1, 38, 1, key="gw_setting")
            st.session_state.current_gameweek = current_gw
            
            your_rank = st.number_input("Your Current Rank (Optional)", 1, 10000000, 1000000, key="rank_setting")
            st.session_state.current_rank = your_rank
            
            st.info(f"""
            **Chat Features:**
            - {'🟢 EuriAI Enhanced' if euriai_key else '🟡 Basic Responses'}
            - Contextual gameweek advice
            - Historical data insights
            - Formation recommendations
            - Transfer strategy guidance
            """)
    
    elif page == "Chip Strategy":
        st.header("💎 Enhanced Chip Usage Strategy")
        
        current_gw = st.number_input("Current Gameweek", 1, 38, 1)
        
        # Enhanced chip strategy with AI insights
        chip_strategy = ChipStrategy()
        recommendations = chip_strategy.recommend_chip_usage(current_gw)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if recommendations:
                st.success("💎 Chip recommendations for this gameweek:")
                for chip, advice in recommendations.items():
                    st.info(f"**{chip}**: {advice}")
            else:
                st.info("No specific chip recommendations for this gameweek.")
            
            # Get AI advice for chip strategy
            chip_query = f"What's the best chip strategy for gameweek {current_gw}?"
            ai_chip_advice = chatbot.get_contextual_response(chip_query, {'gameweek': current_gw})
            
            st.subheader("🤖 AI Chip Strategy Advice")
            st.markdown(ai_chip_advice)
        
        with col2:
            # Display season-long chip strategy
            st.subheader("🗓️ Season-Long Strategy")
            
            chip_timeline = {
                "Wildcard 1": {"gws": "7-10", "reason": "Team assessment after initial weeks", "priority": "Medium"},
                "Free Hit": {"gws": "18, 29", "reason": "Blank gameweeks with few fixtures", "priority": "High"},
                "Wildcard 2": {"gws": "19-25", "reason": "Prepare for double gameweeks", "priority": "High"},
                "Bench Boost": {"gws": "26, 34", "reason": "Double gameweeks with full bench", "priority": "High"},
                "Triple Captain": {"gws": "26, 35", "reason": "Premium player in double gameweek", "priority": "Very High"},
            }
            
            for chip, info in chip_timeline.items():
                with st.container():
                    priority_color = {
                        "Very High": "🔴",
                        "High": "🟠", 
                        "Medium": "🟡",
                        "Low": "🟢"
                    }
                    
                    st.markdown(f"""
                    **{chip}** {priority_color[info['priority']]}
                    - **Best GWs**: {info['gws']}
                    - **Reason**: {info['reason']}
                    - **Priority**: {info['priority']}
                    """)
                    st.write("---")
    
    elif page == "Player Comparison":
        st.header("⚖️ Enhanced Player Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            player1 = st.selectbox("Select Player 1:", players_df['web_name'].unique())
        with col2:
            player2 = st.selectbox("Select Player 2:", players_df['web_name'].unique())
        
        if player1 and player2 and player1 != player2:
            p1_data = players_df[players_df['web_name'] == player1].iloc[0]
            p2_data = players_df[players_df['web_name'] == player2].iloc[0]
            
            # Enhanced comparison metrics
            metrics = [
                'total_points', 'value', 'points_per_million', 'form_float', 
                'minutes', 'starting_xi_probability', 'consistency_score', 'historical_score'
            ]
            metric_labels = [
                'Total Points', 'Price (£M)', 'Points/£M', 'Form', 
                'Minutes', 'Starting XI Prob', 'Consistency', 'Historical Score'
            ]
            
            comparison_df = pd.DataFrame({
                player1: [p1_data[m] for m in metrics],
                player2: [p2_data[m] for m in metrics]
            }, index=metric_labels)
            
            st.subheader("📈 Enhanced Statistical Comparison")
            st.dataframe(comparison_df.style.highlight_max(axis=1))
            
            # Get AI comparison
            comparison_query = f"Compare {player1} vs {player2} for FPL. Who is the better pick?"
            comparison_context = {
                'player1': player1,
                'player2': player2,
                'p1_points': p1_data['total_points'],
                'p2_points': p2_data['total_points'],
                'p1_price': p1_data['value'],
                'p2_price': p2_data['value']
            }
            
            ai_comparison = chatbot.get_contextual_response(comparison_query, comparison_context)
            
            st.subheader("🤖 AI Comparison Analysis")
            st.markdown(ai_comparison)
            
            # Enhanced radar chart
            fig = go.Figure()
            
            # Normalize values for radar chart
            normalized_p1 = []
            normalized_p2 = []
            
            for metric in metrics:
                max_val = players_df[metric].max()
                min_val = players_df[metric].min()
                if max_val != min_val:
                    p1_norm = (p1_data[metric] - min_val) / (max_val - min_val)
                    p2_norm = (p2_data[metric] - min_val) / (max_val - min_val)
                else:
                    p1_norm = p2_norm = 0.5
                normalized_p1.append(p1_norm)
                normalized_p2.append(p2_norm)
            
            fig.add_trace(go.Scatterpolar(
                r=normalized_p1,
                theta=metric_labels,
                fill='toself',
                name=player1,
                line_color='#37003c'
            ))
            
            fig.add_trace(go.Scatterpolar(
                r=normalized_p2,
                theta=metric_labels,
                fill='toself',
                name=player2,
                line_color='#563d7c'
            ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                title="Enhanced Player Comparison Radar Chart"
            )
            
            st.plotly_chart(fig, use_container_width=True)

class ChipStrategy:
    """Enhanced chip strategy with AI recommendations"""
    
    @staticmethod
    def recommend_chip_usage(gameweek: int, fixtures_data: Dict = None) -> Dict:
        """Enhanced chip recommendations"""
        
        recommendations = {
            'Triple Captain': {
                'recommended_gw': [1, 19, 25, 26, 35],
                'description': 'Use on premium players during double gameweeks or excellent fixtures'
            },
            'Bench Boost': {
                'recommended_gw': [26, 34, 37],
                'description': 'Use when your entire bench has favorable fixtures'
            },
            'Free Hit': {
                'recommended_gw': [18, 29, 33],
                'description': 'Use during blank gameweeks when many top players dont play'
            },
            'Wildcard': {
                'recommended_gw': [8, 20, 31],
                'description': 'Use during international breaks or before fixture swings'
            }
        }
        
        current_recommendations = {}
        for chip, info in recommendations.items():
            if gameweek in info['recommended_gw']:
                current_recommendations[chip] = f"Consider using {chip}: {info['description']}"
        
        return current_recommendations

if __name__ == "__main__":
    main()