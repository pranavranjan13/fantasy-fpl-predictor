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

def safe_int(value, default=0):
    """Safely convert a value to int"""
    try:
        if pd.isna(value) or value == '' or value is None:
            return default
        return int(float(value))
    except (ValueError, TypeError):
        return default

class FPLDataManager:
    """FPL data manager"""
    
    def __init__(self):
        self.base_url = "https://fantasy.premierleague.com/api/"
        self.current_season_data = None
        
    def fetch_current_season_data(self):
        """Fetch current season data from FPL API"""
        try:
            response = requests.get(f"{self.base_url}bootstrap-static/", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API returned status code: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"Error fetching FPL data: {str(e)}")
            return None
    
    def get_players_dataframe(self):
        """Get player data as DataFrame"""
        if self.current_season_data is None:
            with st.spinner("Fetching FPL data..."):
                self.current_season_data = self.fetch_current_season_data()
        
        if self.current_season_data:
            players_df = pd.DataFrame(self.current_season_data['elements'])
            
            # Add position names
            position_map = {pos['id']: pos['singular_name'] for pos in self.current_season_data['element_types']}
            players_df['position'] = players_df['element_type'].map(position_map)
            
            # Add team names  
            team_map = {team['id']: team['name'] for team in self.current_season_data['teams']}
            players_df['team_name'] = players_df['team'].map(team_map)
            
            # Calculate key metrics safely
            players_df['value'] = players_df['now_cost'].apply(lambda x: safe_float(x) / 10.0)
            players_df['form_float'] = players_df['form'].apply(safe_float)
            
            # Calculate points per million
            players_df['points_per_million'] = players_df.apply(
                lambda row: safe_float(row['total_points']) / max(safe_float(row['value']), 0.1), axis=1
            )
            
            # Calculate predicted points (simple model)
            players_df['predicted_points'] = players_df.apply(self._calculate_predicted_points, axis=1)
            
            # Starting XI probability
            players_df['starting_xi_probability'] = players_df.apply(self._calculate_starting_probability, axis=1)
            
            # Convert numeric columns
            numeric_cols = ['total_points', 'points_per_game', 'minutes', 'goals_scored', 'assists']
            for col in numeric_cols:
                if col in players_df.columns:
                    players_df[col] = players_df[col].apply(safe_float)
            
            return players_df
        
        return pd.DataFrame()
    
    def _calculate_predicted_points(self, player):
        """Calculate predicted points for a player"""
        total_points = safe_float(player.get('total_points', 0))
        form = safe_float(player.get('form', 0))
        minutes = safe_float(player.get('minutes', 0))
        
        # Simple prediction based on current performance
        base_prediction = total_points * 0.4
        form_factor = form * 2
        minutes_factor = min(minutes / 100, 10)  # Cap minutes influence
        
        return base_prediction + form_factor + minutes_factor
    
    def _calculate_starting_probability(self, player):
        """Calculate starting XI probability"""
        minutes = safe_float(player.get('minutes', 0))
        
        if minutes > 2500:
            return 0.9
        elif minutes > 2000:
            return 0.8
        elif minutes > 1500:
            return 0.7
        elif minutes > 1000:
            return 0.5
        else:
            return 0.3

class FPLPredictor:
    """Simple FPL predictor"""
    
    def __init__(self, players_df):
        self.players_df = players_df
        self.position_limits = {"Goalkeeper": 2, "Defender": 5, "Midfielder": 5, "Forward": 3}
        
    def optimize_team_selection(self, budget=100.0):
        """Generate optimal team within budget"""
        if self.players_df.empty:
            return {"error": "No player data available"}
        
        selected_players = []
        remaining_budget = budget
        
        # Select players by position
        for position in ["Goalkeeper", "Defender", "Midfielder", "Forward"]:
            limit = self.position_limits[position]
            pos_players = self.players_df[self.players_df['position'] == position].copy()
            
            # Filter by budget and sort by predicted points
            affordable = pos_players[pos_players['value'] <= remaining_budget]
            if affordable.empty:
                continue
                
            affordable = affordable.sort_values('predicted_points', ascending=False)
            
            # Select top players for this position
            count = 0
            for _, player in affordable.iterrows():
                if count >= limit:
                    break
                if player['value'] <= remaining_budget:
                    selected_players.append(player.to_dict())
                    remaining_budget -= player['value']
                    count += 1
        
        total_cost = sum(p['value'] for p in selected_players)
        
        return {
            "team": selected_players,
            "total_cost": total_cost,
            "remaining_budget": budget - total_cost
        }
    
    def suggest_starting_eleven(self, team_players=None):
        """Suggest optimal starting XI"""
        if team_players is None:
            team_players = self.players_df
        
        if isinstance(team_players, list):
            team_df = pd.DataFrame(team_players)
        else:
            team_df = team_players
        
        if team_df.empty:
            return {"error": "No team data available"}
        
        # Try different formations
        formations = [
            {"name": "3-4-3", "GK": 1, "DEF": 3, "MID": 4, "FWD": 3},
            {"name": "3-5-2", "GK": 1, "DEF": 3, "MID": 5, "FWD": 2},
            {"name": "4-4-2", "GK": 1, "DEF": 4, "MID": 4, "FWD": 2},
            {"name": "4-3-3", "GK": 1, "DEF": 4, "MID": 3, "FWD": 3},
        ]
        
        best_xi = None
        best_score = -1
        
        for formation in formations:
            xi_players = []
            total_score = 0
            
            # Select goalkeeper
            gks = team_df[team_df['position'] == 'Goalkeeper']
            if len(gks) >= formation["GK"]:
                best_gk = gks.nlargest(formation["GK"], 'predicted_points')
                xi_players.extend(best_gk.to_dict('records'))
                total_score += best_gk['predicted_points'].sum()
            
            # Select defenders
            defs = team_df[team_df['position'] == 'Defender']
            if len(defs) >= formation["DEF"]:
                best_defs = defs.nlargest(formation["DEF"], 'predicted_points')
                xi_players.extend(best_defs.to_dict('records'))
                total_score += best_defs['predicted_points'].sum()
            
            # Select midfielders
            mids = team_df[team_df['position'] == 'Midfielder']
            if len(mids) >= formation["MID"]:
                best_mids = mids.nlargest(formation["MID"], 'predicted_points')
                xi_players.extend(best_mids.to_dict('records'))
                total_score += best_mids['predicted_points'].sum()
            
            # Select forwards
            fwds = team_df[team_df['position'] == 'Forward']
            if len(fwds) >= formation["FWD"]:
                best_fwds = fwds.nlargest(formation["FWD"], 'predicted_points')
                xi_players.extend(best_fwds.to_dict('records'))
                total_score += best_fwds['predicted_points'].sum()
            
            # Check if we have a complete XI
            if len(xi_players) == 11 and total_score > best_score:
                best_xi = {
                    "formation": formation,
                    "players": xi_players,
                    "total_predicted_points": total_score
                }
                best_score = total_score
        
        if best_xi:
            # Find captain (highest predicted points)
            xi_df = pd.DataFrame(best_xi["players"])
            captain_idx = xi_df['predicted_points'].idxmax()
            captain = xi_df.loc[captain_idx]
            
            # Find vice captain (second highest)
            vice_captain_idx = xi_df['predicted_points'].nlargest(2).index[1]
            vice_captain = xi_df.loc[vice_captain_idx]
            
            best_xi["captain"] = captain.to_dict()
            best_xi["vice_captain"] = vice_captain.to_dict()
            
            return best_xi
        
        return {"error": "Could not form a valid starting XI"}

class FPLChatBot:
    """Simple FPL chatbot"""
    
    def __init__(self):
        self.responses = {
            "captain": "🎯 **Captain Selection Tips:**\n• Choose players with good fixtures\n• Prioritize consistent performers\n• Consider form over price\n• Avoid rotation risks",
            "transfer": "🔄 **Transfer Strategy:**\n• Don't take hits unnecessarily\n• Plan transfers around fixtures\n• Monitor price changes\n• Bank transfers for double gameweeks",
            "formation": "⚡ **Formation Guide:**\n• 3-5-2: Premium midfielder heavy\n• 3-4-3: High risk, high reward\n• 4-4-2: Balanced and safe\n• 4-3-3: Mix of attack and stability",
            "chip": "💎 **Chip Strategy:**\n• Wildcard: GW8 and GW20\n• Triple Captain: Double gameweeks\n• Bench Boost: When bench has fixtures\n• Free Hit: Blank gameweeks",
            "budget": "💰 **Budget Tips:**\n• Don't overspend on goalkeepers\n• Invest in 2-3 premium players\n• Use 4.5M defender rotation\n• Keep some money in the bank"
        }
    
    def get_response(self, message):
        """Get chatbot response"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['captain', 'c']):
            return self.responses["captain"]
        elif any(word in message_lower for word in ['transfer', 'buy', 'sell']):
            return self.responses["transfer"]
        elif any(word in message_lower for word in ['formation', 'xi', '11']):
            return self.responses["formation"]
        elif any(word in message_lower for word in ['chip', 'wildcard', 'triple']):
            return self.responses["chip"]
        elif any(word in message_lower for word in ['budget', 'money', 'price']):
            return self.responses["budget"]
        else:
            return "🏆 **General FPL Tips:**\n• Follow team news closely\n• Analyze fixtures 3-4 weeks ahead\n• Don't chase last week's points\n• Be patient with your decisions\n• Join FPL communities for insights"

def main():
    """Main application"""
    
    st.markdown('<h1 class="main-header">⚽ FPL AI Assistant</h1>', unsafe_allow_html=True)
    
    # Initialize components
    data_manager = FPLDataManager()
    chatbot = FPLChatBot()
    
    # Sidebar navigation
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
        st.error("❌ Unable to load FPL data. Please check your internet connection and try again.")
        st.info("💡 The app needs to connect to the official FPL API to fetch current player data.")
        return
    
    predictor = FPLPredictor(players_df)
    st.success(f"✅ Loaded data for {len(players_df)} players from FPL API")
    
    if page == "Team Optimization":
        st.header("🎯 Team Optimization")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            budget = st.slider("Budget (£M)", 95.0, 105.0, 100.0, 0.5)
            
            if st.button("🚀 Generate Optimal Team", type="primary"):
                with st.spinner("Optimizing team selection..."):
                    result = predictor.optimize_team_selection(budget)
                    
                    if "error" in result:
                        st.error(f"❌ {result['error']}")
                    else:
                        st.success(f"✅ Generated optimal team! Cost: £{result['total_cost']:.1f}M")
                        
                        # Get starting XI
                        xi_result = predictor.suggest_starting_eleven(result['team'])
                        
                        if "error" not in xi_result:
                            # Display formation
                            formation = xi_result['formation']
                            st.markdown(f'<div class="formation-display">Formation: {formation["name"]}</div>', unsafe_allow_html=True)
                            
                            # Captain info
                            captain = xi_result['captain']
                            vice_captain = xi_result['vice_captain']
                            
                            col_c1, col_c2 = st.columns(2)
                            with col_c1:
                                st.info(f"👑 **Captain:** {captain['web_name']} ({captain['predicted_points']:.1f} pts)")
                            with col_c2:
                                st.info(f"🎖️ **Vice:** {vice_captain['web_name']} ({vice_captain['predicted_points']:.1f} pts)")
                            
                            # Display starting XI
                            st.subheader("🔥 Starting XI")
                            
                            positions = ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']
                            for pos in positions:
                                pos_players = [p for p in xi_result['players'] if p['position'] == pos]
                                if pos_players:
                                    st.write(f"**{pos}s:**")
                                    for player in pos_players:
                                        is_captain = player['web_name'] == captain['web_name']
                                        is_vice = player['web_name'] == vice_captain['web_name']
                                        badge = " (C)" if is_captain else " (VC)" if is_vice else ""
                                        
                                        st.markdown(
                                            f'<div class="starting-xi-card">'
                                            f'<strong>{player["web_name"]}{badge}</strong> ({player["team_name"]})<br>'
                                            f'£{player["value"]:.1f}M • {player["predicted_points"]:.1f} pts • '
                                            f'Form: {player["form_float"]:.1f}'
                                            f'</div>',
                                            unsafe_allow_html=True
                                        )
                            
                            # Show bench players
                            all_team_names = {p['web_name'] for p in xi_result['players']}
                            bench_players = [p for p in result['team'] if p['web_name'] not in all_team_names]
                            
                            if bench_players:
                                st.subheader("🪑 Bench")
                                for i, player in enumerate(bench_players[:4], 1):
                                    st.markdown(
                                        f'<div class="bench-card">'
                                        f'{i}. <strong>{player["web_name"]}</strong> ({player["position"]}) - '
                                        f'£{player["value"]:.1f}M • {player["predicted_points"]:.1f} pts'
                                        f'</div>',
                                        unsafe_allow_html=True
                                    )
                        
                        # Full squad breakdown
                        with st.expander("📋 Full Squad Breakdown"):
                            squad_by_position = {}
                            for player in result['team']:
                                pos = player['position']
                                if pos not in squad_by_position:
                                    squad_by_position[pos] = []
                                squad_by_position[pos].append(player)
                            
                            for pos in positions:
                                if pos in squad_by_position:
                                    st.subheader(f"{pos}s ({len(squad_by_position[pos])})")
                                    for player in squad_by_position[pos]:
                                        st.write(f"• **{player['web_name']}** - £{player['value']:.1f}M ({player['predicted_points']:.1f} pts)")
        
        with col2:
            st.subheader("📊 Quick Stats")
            if not players_df.empty:
                st.metric("Total Players", len(players_df))
                st.metric("Avg Player Price", f"£{players_df['value'].mean():.1f}M")
                st.metric("Top Scorer", f"{players_df.loc[players_df['total_points'].idxmax(), 'web_name']}")
                
                # Top 5 by predicted points
                st.subheader("🌟 Top Performers")
                top_players = players_df.nlargest(5, 'predicted_points')[['web_name', 'position', 'predicted_points', 'value']]
                for _, player in top_players.iterrows():
                    st.write(f"**{player['web_name']}** ({player['position']}) - {player['predicted_points']:.1f} pts")
    
    elif page == "Squad Analysis":
        st.header("📋 Squad Analysis")
        
        st.info("Select your current 15 players to analyze your squad")
        
        # Player selection
        current_squad = []
        
        with st.expander("🔧 Select Your Squad", expanded=True):
            cols = st.columns(3)
            player_names = [''] + sorted(players_df['web_name'].unique())
            
            for i in range(15):
                with cols[i % 3]:
                    player = st.selectbox(f"Player {i+1}", player_names, key=f"squad_player_{i}")
                    if player:
                        current_squad.append(player)
        
        if len(current_squad) >= 11:
            st.success(f"✅ Selected {len(current_squad)} players")
            
            if st.button("🔍 Analyze My Squad", type="primary"):
                # Get player data for selected squad
                squad_data = players_df[players_df['web_name'].isin(current_squad)].copy()
                
                if not squad_data.empty:
                    # Get optimal XI from current squad
                    xi_result = predictor.suggest_starting_eleven(squad_data)
                    
                    if "error" not in xi_result:
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.subheader("🎯 Recommended Starting XI")
                            
                            formation = xi_result['formation']
                            st.markdown(f'<div class="formation-display">Formation: {formation["name"]}</div>', unsafe_allow_html=True)
                            
                            # Captain suggestions
                            captain = xi_result['captain']
                            vice_captain = xi_result['vice_captain']
                            
                            st.info(f"👑 **Recommended Captain:** {captain['web_name']} ({captain['predicted_points']:.1f} pts)")
                            st.info(f"🎖️ **Vice-Captain:** {vice_captain['web_name']} ({vice_captain['predicted_points']:.1f} pts)")
                            
                            # Show starting XI by position
                            for pos in ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']:
                                pos_players = [p for p in xi_result['players'] if p['position'] == pos]
                                if pos_players:
                                    st.write(f"**{pos}s:**")
                                    for player in pos_players:
                                        is_captain = player['web_name'] == captain['web_name']
                                        is_vice = player['web_name'] == vice_captain['web_name']
                                        badge = " (C)" if is_captain else " (VC)" if is_vice else ""
                                        
                                        st.markdown(
                                            f'<div class="starting-xi-card">'
                                            f'<strong>{player["web_name"]}{badge}</strong> ({player["team_name"]})<br>'
                                            f'£{player["value"]:.1f}M • Predicted: {player["predicted_points"]:.1f} pts'
                                            f'</div>',
                                            unsafe_allow_html=True
                                        )
                        
                        with col2:
                            st.subheader("📊 Squad Statistics")
                            
                            total_value = squad_data['value'].sum()
                            total_points = xi_result['total_predicted_points']
                            
                            st.metric("Squad Value", f"£{total_value:.1f}M")
                            st.metric("XI Predicted Points", f"{total_points:.1f}")
                            st.metric("Players Selected", len(current_squad))
                            
                            # Position breakdown
                            st.subheader("Position Breakdown")
                            pos_counts = squad_data['position'].value_counts()
                            st.bar_chart(pos_counts)
                    else:
                        st.error("❌ Could not form a starting XI from your selected players")
                else:
                    st.error("❌ No valid players found in your selection")
        else:
            st.warning(f"⚠️ Please select at least 11 players (currently selected: {len(current_squad)})")
    
    elif page == "FPL Strategy Chat":
        st.header("💬 FPL Strategy Chat")
        
        # Initialize session state
        if 'chat_messages' not in st.session_state:
            st.session_state.chat_messages = [
                {"role": "assistant", "content": "👋 Hello! I'm your FPL Strategy Assistant. Ask me about captains, transfers, formations, chips, or general FPL advice!"}
            ]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display chat messages
            for message in st.session_state.chat_messages:
                if message["role"] == "user":
                    st.markdown(f"**You:** {message['content']}")
                else:
                    st.markdown(f"**🤖 FPL Bot:** {message['content']}")
                st.markdown("---")
            
            # User input
            user_input = st.text_input("Ask me anything about FPL:", placeholder="e.g., Who should I captain this week?")
            
            col_send, col_clear = st.columns([1, 1])
            
            with col_send:
                if st.button("Send 📤", type="primary") and user_input:
                    # Get response
                    response = chatbot.get_response(user_input)
                    
                    # Add to chat history
                    st.session_state.chat_messages.append({"role": "user", "content": user_input})
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})
                    
                    st.rerun()
            
            with col_clear:
                if st.button("Clear Chat 🗑️"):
                    st.session_state.chat_messages = [
                        {"role": "assistant", "content": "👋 Chat cleared! How can I help you with FPL?"}
                    ]
                    st.rerun()
        
        with col2:
            st.subheader("💡 Quick Questions")
            
            quick_questions = [
                "Who should I captain?",
                "When to use wildcard?", 
                "Best formation?",
                "Transfer strategy?",
                "Chip timing advice?"
            ]
            
            for question in quick_questions:
                if st.button(f"❓ {question}", key=f"q_{question}"):
                    response = chatbot.get_response(question)
                    st.session_state.chat_messages.append({"role": "user", "content": question})
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})
                    st.rerun()
    
    elif page == "Player Comparison":
        st.header("⚖️ Player Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            player1 = st.selectbox("Select Player 1:", players_df['web_name'].unique())
        
        with col2:
            player2 = st.selectbox("Select Player 2:", players_df['web_name'].unique())
        
        if player1 and player2 and player1 != player2:
            p1_data = players_df[players_df['web_name'] == player1].iloc[0]
            p2_data = players_df[players_df['web_name'] == player2].iloc[0]
            
            st.subheader("📊 Comparison")
            
            # Create comparison table
            comparison_data = {
                'Metric': ['Total Points', 'Price (£M)', 'Points per £M', 'Form', 'Minutes', 'Goals', 'Assists', 'Predicted Points'],
                player1: [
                    p1_data['total_points'],
                    p1_data['value'],
                    p1_data['points_per_million'],
                    p1_data['form_float'],
                    p1_data['minutes'],
                    p1_data.get('goals_scored', 0),
                    p1_data.get('assists', 0),
                    p1_data['predicted_points']
                ],
                player2: [
                    p2_data['total_points'],
                    p2_data['value'], 
                    p2_data['points_per_million'],
                    p2_data['form_float'],
                    p2_data['minutes'],
                    p2_data.get('goals_scored', 0),
                    p2_data.get('assists', 0),
                    p2_data['predicted_points']
                ]
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Recommendation
            if p1_data['predicted_points'] > p2_data['predicted_points']:
                st.success(f"🎯 **Recommendation:** {player1} has higher predicted points ({p1_data['predicted_points']:.1f} vs {p2_data['predicted_points']:.1f})")
            elif p2_data['predicted_points'] > p1_data['predicted_points']:
                st.success(f"🎯 **Recommendation:** {player2} has higher predicted points ({p2_data['predicted_points']:.1f} vs {p1_data['predicted_points']:.1f})")
            else:
                st.info("📊 Both players have similar predicted points")
            
            # Value analysis
            if p1_data['points_per_million'] > p2_data['points_per_million']:
                st.info(f"💰 **Better Value:** {player1} offers better points per million")
            else:
                st.info(f"💰 **Better Value:** {player2} offers better points per million")

if __name__ == "__main__":
    main()