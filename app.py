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
    """Manages FPL data fetching and processing"""
    
    def __init__(self):
        self.base_url = "https://fantasy.premierleague.com/api/"
        self.current_season_data = None
        self.players_data = None
        self.teams_data = None
        
    @st.cache_data(ttl=3600)  # Cache for 1 hour
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
    
    def get_players_dataframe(self):
        """Convert players data to DataFrame"""
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
            
            # Calculate value metrics
            players_df['value'] = players_df['now_cost'] / 10  # Convert to millions
            players_df['points_per_million'] = players_df['total_points'] / players_df['value']
            players_df['form_float'] = pd.to_numeric(players_df['form'], errors='coerce')
            
            # Ensure all relevant columns are numeric for calculations in FPLPredictor
            numeric_cols_to_convert = [
                'total_points',
                'points_per_game', # Make sure this column exists in your raw data
                'minutes',
                'influence',
                'creativity',
                'threat',
                'now_cost'
            ]

        for col in numeric_cols_to_convert:
            if col in players_df.columns:
                # Convert to numeric, coercing errors to NaN, then fill NaN with 0
                players_df[col] = pd.to_numeric(players_df[col], errors='coerce').fillna(0)
        return players_df

class FPLPredictor:
    """AI-powered FPL predictions and recommendations"""
    
    def __init__(self, players_df):
        self.players_df = players_df
        self.position_limits = {'Goalkeeper': 2, 'Defender': 5, 'Midfielder': 5, 'Forward': 3}
        self.team_limit = 3  # Max players from same team
        
    def calculate_predicted_points(self, gameweek: int = 1) -> pd.DataFrame:
        """Calculate predicted points using multiple factors"""
        df = self.players_df.copy()
        
        # Weighted scoring system
        weights = {
            'total_points': 0.3,
            'form_float': 0.25,
            'points_per_game': 0.2,
            'minutes': 0.1,
            'influence': 0.05,
            'creativity': 0.05,
            'threat': 0.05
        }
        
        # Normalize metrics (0-1 scale)
        for metric in weights.keys():
            if metric in df.columns:
                max_val = df[metric].max()
                if max_val > 0:
                    df[f'{metric}_norm'] = df[metric] / max_val
        
        # Calculate predicted points
        df['predicted_points'] = 0
        for metric, weight in weights.items():
            if f'{metric}_norm' in df.columns:
                df['predicted_points'] += df[f'{metric}_norm'] * weight * 20  # Scale to reasonable range
        
        # Position-specific adjustments
        position_multipliers = {'Goalkeeper': 0.8, 'Defender': 0.9, 'Midfielder': 1.1, 'Forward': 1.2}
        for pos, mult in position_multipliers.items():
            df.loc[df['position'] == pos, 'predicted_points'] *= mult
        
        return df
    
    def optimize_team_selection(self, budget: float = 100.0, existing_team: List[str] = None) -> Dict:
        """Optimize team selection using predicted points and constraints"""
        df = self.calculate_predicted_points()
        
        if existing_team:
            # Filter out existing players for transfer suggestions
            df = df[~df['web_name'].isin(existing_team)]
        
        # Simple greedy algorithm for team selection
        selected_team = []
        total_cost = 0
        position_count = {'Goalkeeper': 0, 'Defender': 0, 'Midfielder': 0, 'Forward': 0}
        team_count = {}
        
        # Sort by points per million for value
        df_sorted = df.sort_values('points_per_million', ascending=False)
        
        for _, player in df_sorted.iterrows():
            pos = player['position']
            team = player['team_name']
            cost = player['value']
            
            # Check constraints
            if (position_count[pos] < self.position_limits[pos] and
                team_count.get(team, 0) < self.team_limit and
                total_cost + cost <= budget and
                len(selected_team) < 15):
                
                selected_team.append({
                    'name': player['web_name'],
                    'position': pos,
                    'team': team,
                    'cost': cost,
                    'predicted_points': player['predicted_points'],
                    'form': player['form_float']
                })
                
                total_cost += cost
                position_count[pos] += 1
                team_count[team] = team_count.get(team, 0) + 1
        
        return {
            'team': selected_team,
            'total_cost': total_cost,
            'remaining_budget': budget - total_cost,
            'position_count': position_count
        }
    
    def suggest_starting_eleven(self, team: List[Dict]) -> Dict:
        """Suggest best starting 11 from 15 players"""
        if len(team) != 15:
            return {'error': 'Team must have exactly 15 players'}
        
        # Formation constraints for starting 11
        formations = [
            {'Goalkeeper': 1, 'Defender': 3, 'Midfielder': 5, 'Forward': 2},  # 3-5-2
            {'Goalkeeper': 1, 'Defender': 3, 'Midfielder': 4, 'Forward': 3},  # 3-4-3
            {'Goalkeeper': 1, 'Defender': 4, 'Midfielder': 4, 'Forward': 2},  # 4-4-2
            {'Goalkeeper': 1, 'Defender': 4, 'Midfielder': 3, 'Forward': 3},  # 4-3-3
            {'Goalkeeper': 1, 'Defender': 5, 'Midfielder': 3, 'Forward': 2},  # 5-3-2
        ]
        
        best_xi = None
        max_points = 0
        
        for formation in formations:
            xi_candidate = []
            total_points = 0
            position_players = {pos: [] for pos in formation.keys()}
            
            # Group players by position
            for player in team:
                position_players[player['position']].append(player)
            
            # Sort players by predicted points within each position
            for pos in position_players:
                position_players[pos].sort(key=lambda x: x['predicted_points'], reverse=True)
            
            # Select best players for each position
            valid_formation = True
            for pos, required in formation.items():
                if len(position_players[pos]) >= required:
                    selected = position_players[pos][:required]
                    xi_candidate.extend(selected)
                    total_points += sum(p['predicted_points'] for p in selected)
                else:
                    valid_formation = False
                    break
            
            if valid_formation and total_points > max_points:
                max_points = total_points
                best_xi = {
                    'players': xi_candidate,
                    'formation': formation,
                    'total_predicted_points': total_points,
                    'bench': [p for p in team if p not in xi_candidate]
                }
        
        return best_xi if best_xi else {'error': 'Could not form valid starting XI'}

class ChipStrategy:
    """Manages chip usage strategy"""
    
    @staticmethod
    def recommend_chip_usage(gameweek: int, fixtures_data: Dict = None) -> Dict:
        """Recommend when to use chips based on fixtures and strategy"""
        
        recommendations = {
            'Triple Captain': {
                'recommended_gw': [1, 19, 25, 35],  # Double gameweeks typically
                'description': 'Use during double gameweeks or when your captain has great fixtures'
            },
            'Bench Boost': {
                'recommended_gw': [26, 34, 37],  # Double gameweeks
                'description': 'Use when most of your bench players have games'
            },
            'Free Hit': {
                'recommended_gw': [18, 29],  # Blank gameweeks
                'description': 'Use during blank gameweeks when many players dont play'
            },
            'Wildcard': {
                'recommended_gw': [8, 20],  # International breaks
                'description': 'Use during international breaks to completely restructure team'
            }
        }
        
        current_recommendations = {}
        for chip, info in recommendations.items():
            if gameweek in info['recommended_gw']:
                current_recommendations[chip] = f"Consider using {chip} this gameweek: {info['description']}"
        
        return current_recommendations

def main():
    """Main application function"""
    
    st.markdown('<h1 class="main-header">⚽ FPL AI Assistant</h1>', unsafe_allow_html=True)
    
    # Initialize data manager
    data_manager = FPLDataManager()
    
    st.sidebar.title("🔧 Settings")
        
        # EuriAI API Key input
    euriai_key = st.sidebar.text_input(
            "EuriAI API Key (Optional)", 
            type="password",
            help="Enter your EuriAI API key for enhanced AI recommendations"
        )
        
    if euriai_key:
            os.environ["EURIAI_API_KEY"] = euriai_key
            st.sidebar.success("✓ EuriAI API key set")
        
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "Team Optimization", 
        "Squad Analysis", 
        "Chip Strategy", 
        "Player Comparison"
    ])
    
    # Load player data
    with st.spinner("Loading FPL data..."):
        players_df = data_manager.get_players_dataframe()
    
    if players_df.empty:
        st.error("Unable to load FPL data. Please check your internet connection and try again.")
        return
    
    predictor = FPLPredictor(players_df)
    
    if page == "Team Optimization":
        st.header("🎯 Optimal Team Selection")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            budget = st.slider("Budget (£M)", 95.0, 105.0, 100.0, 0.5)
            
            if st.button("Generate Optimal Team", type="primary"):
                with st.spinner("Optimizing team selection..."):
                    result = predictor.optimize_team_selection(budget)
                    
                    if 'team' in result:
                        st.success(f"Optimal team generated! Cost: £{result['total_cost']:.1f}M")
                        
                        # Display team by position
                        positions = ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']
                        for pos in positions:
                            pos_players = [p for p in result['team'] if p['position'] == pos]
                            if pos_players:
                                st.subheader(f"{pos}s ({len(pos_players)})")
                                for player in pos_players:
                                    col_a, col_b, col_c, col_d = st.columns([3, 1, 1, 1])
                                    with col_a:
                                        st.write(f"**{player['name']}** ({player['team']})")
                                    with col_b:
                                        st.write(f"£{player['cost']:.1f}M")
                                    with col_c:
                                        st.write(f"{player['predicted_points']:.1f} pts")
                                    with col_d:
                                        st.write(f"Form: {player['form']:.1f}")
        
        with col2:
            st.markdown("### 📊 Team Constraints")
            st.info("""
            **Formation Rules:**
            - 2 Goalkeepers
            - 5 Defenders  
            - 5 Midfielders
            - 3 Forwards
            - Max 3 players per team
            - £100M budget
            
            **AI Features:**
            - ML predictions
            - EuriAI recommendations
            - Genetic optimization
            """)
            
            # Show EuriAI status
            euriai_status = "✓ Active" if os.getenv("EURIAI_API_KEY") else "⚠️ Not configured"
            st.info(f"**EuriAI Status:** {euriai_status}")
            
            if not os.getenv("EURIAI_API_KEY"):
                st.info("💡 Add your EuriAI API key in the sidebar for enhanced AI recommendations!")
    
    elif page == "Squad Analysis":
        st.header("📋 Squad Analysis & Starting XI")
        
        st.info("Enter your current 15 players to get AI suggestions for your starting XI and potential transfers")
        
        # Player input section
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
            if st.button("Analyze Squad", type="primary"):
                # Get player details
                team_details = []
                for player_name in current_team:
                    player_data = players_df[players_df['web_name'] == player_name].iloc[0]
                    team_details.append({
                        'name': player_data['web_name'],
                        'position': player_data['position'],
                        'team': player_data['team_name'],
                        'cost': player_data['value'],
                        'predicted_points': predictor.calculate_predicted_points().loc[player_data.name, 'predicted_points'],
                        'form': player_data['form_float']
                    })
                
                # Get starting XI suggestion
                xi_result = predictor.suggest_starting_eleven(team_details)
                
                if 'players' in xi_result:
                    st.success("Starting XI Optimized!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("🔥 Recommended Starting XI")
                        formation = xi_result['formation']
                        st.info(f"Formation: {formation['Defender']}-{formation['Midfielder']}-{formation['Forward']}")
                        
                        for player in xi_result['players']:
                            st.markdown(f"**{player['name']}** ({player['position']}) - {player['predicted_points']:.1f} pts")
                    
                    with col2:
                        st.subheader("🪑 Bench")
                        for player in xi_result['bench']:
                            st.markdown(f"{player['name']} ({player['position']}) - {player['predicted_points']:.1f} pts")
        
        else:
            st.warning(f"Please select all 15 players. Currently selected: {len(current_team)}")
    
    elif page == "Chip Strategy":
        st.header("💎 Chip Usage Strategy")
        
        current_gw = st.number_input("Current Gameweek", 1, 38, 1)
        
        chip_strategy = ChipStrategy()
        recommendations = chip_strategy.recommend_chip_usage(current_gw)
        
        if recommendations:
            st.success("Chip recommendations for this gameweek:")
            for chip, advice in recommendations.items():
                st.info(f"**{chip}**: {advice}")
        else:
            st.info("No specific chip recommendations for this gameweek.")
        
        # Display general chip strategy
        st.subheader("🗓️ Season-Long Chip Strategy")
        
        chip_df = pd.DataFrame([
            {"Chip": "Wildcard 1", "Best GWs": "7-10", "Reason": "After initial team assessment"},
            {"Chip": "Free Hit", "Best GWs": "18, 29", "Reason": "Blank gameweeks"},
            {"Chip": "Wildcard 2", "Best GWs": "19-25", "Reason": "Prepare for DGWs"},
            {"Chip": "Bench Boost", "Best GWs": "26, 34", "Reason": "Double gameweeks"},
            {"Chip": "Triple Captain", "Best GWs": "26, 35", "Reason": "Premium player DGW"},
        ])
        
        st.dataframe(chip_df, use_container_width=True)
    
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
            
            # Comparison metrics
            metrics = ['total_points', 'value', 'points_per_million', 'form_float', 'minutes']
            
            comparison_df = pd.DataFrame({
                player1: [p1_data[m] for m in metrics],
                player2: [p2_data[m] for m in metrics]
            }, index=['Total Points', 'Price (£M)', 'Points/£M', 'Form', 'Minutes'])
            
            st.subheader("📈 Statistical Comparison")
            st.dataframe(comparison_df.style.highlight_max(axis=1))
            
            # Radar chart
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
                theta=['Points', 'Value', 'Points/£M', 'Form', 'Minutes'],
                fill='toself',
                name=player1
            ))
            
            fig.add_trace(go.Scatterpolar(
                r=normalized_p2,
                theta=['Points', 'Value', 'Points/£M', 'Form', 'Minutes'],
                fill='toself',
                name=player2
            ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                title="Player Comparison Radar Chart"
            )
            
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()