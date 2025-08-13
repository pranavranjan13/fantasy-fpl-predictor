import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime
import time

# Import our modules (we'll embed them in the same file for simplicity)
from typing import List, Dict, Optional, Tuple
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Embed the core classes here for Streamlit Cloud compatibility
# [We'll include simplified versions of the key classes]

st.set_page_config(
    page_title="Fantasy Football Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("⚽ Fantasy Football Team Predictor")
    st.markdown("AI-powered team selection and strategy optimization")
    
    # Add GitHub link
    st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-View%20Source-black?logo=github)](https://github.com/pranavranjan13/fantasy-fpl-predictor)")
    
    # Rest of your Streamlit app code...
# Configuration
API_BASE_URL = "http://localhost:8000/api"

def main():
    st.set_page_config(
        page_title="Fantasy Football Predictor",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("⚽ Fantasy Football Team Predictor")
    st.markdown("AI-powered team selection and strategy optimization")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        budget = st.slider("Budget (£m)", 95.0, 105.0, 100.0, 0.1)
        gameweek = st.number_input("Target Gameweek", 1, 38, 1)
        
        st.subheader("Team Constraints")
        max_per_team = st.selectbox("Max players per team", [3, 2, 1], 0)
        
        # Chip selection
        st.subheader("Available Chips")
        available_chips = st.multiselect(
            "Chips not yet used",
            ["wildcard", "free_hit", "bench_boost", "triple_captain"],
            default=["wildcard", "free_hit", "bench_boost", "triple_captain"]
        )
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Team Prediction", "Player Analysis", "Chip Strategy", "Performance Dashboard"])
    
    with tab1:
        st.header("Optimal Team Selection")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("Generate Optimal Team", type="primary"):
                with st.spinner("Analyzing players and optimizing team..."):
                    try:
                        # Make API call to get team prediction
                        response = requests.post(
                            f"{API_BASE_URL}/predict-team",
                            json={
                                "gameweek": gameweek,
                                "budget": budget,
                                "constraints": {
                                    "total_budget": budget,
                                    "max_players_per_team": max_per_team
                                }
                            }
                        )
                        
                        if response.status_code == 200:
                            team_data = response.json()
                            display_team_selection(team_data)
                        else:
                            st.error("Failed to generate team prediction")
                    
                    except Exception as e:
                        st.error(f"Connection error: {str(e)}")
                        # Display sample data for demo
                        display_sample_team()
        
        with col2:
            st.info("💡 **Tips:**\n- Consider upcoming fixtures\n- Monitor price changes\n- Check injury news\n- Plan transfers ahead")
            
            # Strategic advice section
            st.subheader("Ask for Strategic Advice")
            advice_query = st.text_input(
                "Ask a question about your FPL strategy:",
                placeholder="e.g., Should I captain Haaland this week?"
            )
            
            if st.button("Get AI Advice") and advice_query:
                with st.spinner("Getting strategic advice..."):
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/strategic-advice",
                            params={"query": advice_query}
                        )
                        if response.status_code == 200:
                            advice = response.json()["advice"]
                            st.success("🤖 **AI Strategic Advice:**")
                            st.write(advice)
                        else:
                            st.error("Could not get strategic advice")
                    except Exception as e:
                        st.warning("Strategic advice service unavailable. Make sure EURI_API_KEY is set.")
    
    with tab2:
        st.header("Player Performance Analysis")
        
        # Player comparison tool
        col1, col2 = st.columns(2)
        
        with col1:
            selected_players = st.multiselect(
                "Select players to compare",
                ["Haaland", "Salah", "Kane", "Son", "De Bruyne"]  # Sample players
            )
        
        with col2:
            metrics = st.multiselect(
                "Metrics to analyze",
                ["Points", "Price", "Form", "ICT Index", "Expected Goals"],
                default=["Points", "Form"]
            )
        
        if selected_players and metrics:
            display_player_comparison(selected_players, metrics)
    
    with tab3:
        st.header("Chip Strategy Recommendations")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("Get Chip Recommendations"):
                with st.spinner("Analyzing fixtures and generating recommendations..."):
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/chip-recommendation",
                            params={
                                "gameweek": gameweek,
                                "available_chips": available_chips
                            }
                        )
                        if response.status_code == 200:
                            display_chip_recommendations(gameweek, available_chips)
                        else:
                            st.error("Could not get chip recommendations")
                    except Exception as e:
                        display_chip_recommendations(gameweek, available_chips)  # Fallback to sample data
        
        with col2:
            st.info("🎯 **Chip Usage Tips:**\n- Wildcard: Major team overhauls\n- Free Hit: Blank gameweeks\n- Bench Boost: Double gameweeks\n- Triple Captain: Premium DGW players")
    
    with tab4:
        st.header("Performance Dashboard")
        display_performance_dashboard()

def display_team_selection(team_data):
    """Display the optimal team selection"""
    st.success(f"Optimal team generated! Predicted points: {team_data['predicted_points']:.1f}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Cost", f"£{team_data['total_cost']:.1f}m")
    
    with col2:
        st.metric("Predicted Points", f"{team_data['predicted_points']:.1f}")
    
    with col3:
        budget_remaining = 100.0 - team_data['total_cost']
        st.metric("Budget Remaining", f"£{budget_remaining:.1f}m")
    
    # Display team formation
    players_df = pd.DataFrame([p for p in team_data['players']])
    
    st.subheader("Starting XI")
    starting_players = players_df[players_df['id'].isin(team_data['starting_eleven'])]
    
    # Group by position for display
    for position in ['GKP', 'DEF', 'MID', 'FWD']:
        pos_players = starting_players[starting_players['position'] == position]
        if not pos_players.empty:
            st.write(f"**{position}:**")
            for _, player in pos_players.iterrows():
                captain_emoji = "🔥" if player['id'] == team_data['captain'] else "🅲" if player['id'] == team_data['vice_captain'] else ""
                st.write(f"- {player['name']} ({player['team']}) - £{player['price']:.1f}m {captain_emoji}")
    
    st.subheader("Bench")
    bench_players = players_df[players_df['id'].isin(team_data['bench'])]
    for _, player in bench_players.iterrows():
        st.write(f"- {player['name']} ({player['team']}) - £{player['price']:.1f}m")

def display_sample_team():
    """Display sample team for demo purposes"""
    st.success("Sample optimal team generated! Predicted points: 65.4")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Cost", "£99.2m")
    
    with col2:
        st.metric("Predicted Points", "65.4")
    
    with col3:
        st.metric("Budget Remaining", "£0.8m")
    
    st.subheader("Starting XI")
    
    # Sample team display
    positions = {
        "GKP": ["Alisson (Liverpool) - £5.5m 🔥"],
        "DEF": [
            "Alexander-Arnold (Liverpool) - £7.2m",
            "Saliba (Arsenal) - £5.1m", 
            "Trippier (Newcastle) - £6.0m"
        ],
        "MID": [
            "Salah (Liverpool) - £12.8m 🅲",
            "De Bruyne (Man City) - £11.4m",
            "Saka (Arsenal) - £9.2m",
            "Martinelli (Arsenal) - £6.8m"
        ],
        "FWD": [
            "Haaland (Man City) - £14.1m",
            "Kane (Tottenham) - £11.3m",
            "Mitrovic (Fulham) - £6.7m"
        ]
    }
    
    for pos, players in positions.items():
        st.write(f"**{pos}:**")
        for player in players:
            st.write(f"- {player}")
    
    st.subheader("Bench")
    bench = [
        "Raya (Arsenal) - £4.5m",
        "Dunk (Brighton) - £4.8m", 
        "Andreas (Fulham) - £4.5m",
        "Archer (Aston Villa) - £4.5m"
    ]
    
    for player in bench:
        st.write(f"- {player}")

def display_player_comparison(selected_players, metrics):
    """Display player comparison charts"""
    
    # Sample data for comparison
    sample_data = {
        "Haaland": {"Points": 224, "Price": 14.1, "Form": 8.2, "ICT Index": 18.4, "Expected Goals": 0.89},
        "Salah": {"Points": 196, "Price": 12.8, "Form": 6.8, "ICT Index": 16.2, "Expected Goals": 0.72},
        "Kane": {"Points": 188, "Price": 11.3, "Form": 7.1, "ICT Index": 15.8, "Expected Goals": 0.68},
        "Son": {"Points": 134, "Price": 9.8, "Form": 5.2, "ICT Index": 12.1, "Expected Goals": 0.51},
        "De Bruyne": {"Points": 176, "Price": 11.4, "Form": 7.8, "ICT Index": 17.2, "Expected Goals": 0.34}
    }
    
    comparison_data = []
    for player in selected_players:
        if player in sample_data:
            row = {"Player": player}
            row.update(sample_data[player])
            comparison_data.append(row)
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        
        # Create subplots for each metric
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=metrics[:4] if len(metrics) >= 4 else metrics,
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        colors = px.colors.qualitative.Set3
        
        for i, metric in enumerate(metrics[:4]):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            fig.add_trace(
                go.Bar(
                    x=df["Player"],
                    y=df[metric],
                    name=metric,
                    marker_color=colors[i % len(colors)],
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig.update_layout(height=600, title_text="Player Comparison")
        st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        st.subheader("Detailed Comparison")
        display_columns = ["Player"] + metrics
        st.dataframe(df[display_columns], use_container_width=True)

def display_chip_recommendations(gameweek, available_chips):
    """Display chip strategy recommendations"""
    
    # Sample chip recommendations
    recommendations = [
        {
            "chip": "Triple Captain",
            "recommended_gameweek": gameweek + 3,
            "confidence": 0.85,
            "reasoning": "Haaland has a double gameweek against Burnley (H) and Sheffield United (A) - both teams have poor defensive records",
            "expected_benefit": 18.2
        },
        {
            "chip": "Bench Boost", 
            "recommended_gameweek": gameweek + 5,
            "confidence": 0.72,
            "reasoning": "Multiple teams have double gameweeks, allowing bench players to contribute significantly",
            "expected_benefit": 12.8
        },
        {
            "chip": "Wildcard",
            "recommended_gameweek": gameweek + 1,
            "confidence": 0.68,
            "reasoning": "International break provides good opportunity for major squad overhaul before favorable fixture run",
            "expected_benefit": 15.4
        },
        {
            "chip": "Free Hit",
            "recommended_gameweek": gameweek + 8,
            "confidence": 0.91,
            "reasoning": "Many premium teams blank due to cup fixtures - Free Hit allows full team of double gameweek players",
            "expected_benefit": 22.6
        }
    ]
    
    for rec in recommendations:
        if rec["chip"].lower().replace(" ", "_") in available_chips:
            
            # Confidence color coding
            if rec["confidence"] >= 0.8:
                confidence_color = "🟢"
            elif rec["confidence"] >= 0.6:
                confidence_color = "🟡"
            else:
                confidence_color = "🔴"
            
            with st.container():
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.subheader(f"{rec['chip']} {confidence_color}")
                    st.write(rec["reasoning"])
                
                with col2:
                    st.metric("Recommended GW", rec["recommended_gameweek"])
                    st.metric("Confidence", f"{rec['confidence']:.0%}")
                
                with col3:
                    st.metric("Expected Benefit", f"+{rec['expected_benefit']:.1f} pts")
                
                st.divider()

def display_performance_dashboard():
    """Display performance analytics dashboard"""
    
    # Sample historical performance data
    gameweeks = list(range(1, 15))
    team_points = [52, 48, 71, 38, 65, 84, 45, 73, 56, 61, 78, 42, 69, 58]
    average_points = [54, 51, 62, 43, 58, 72, 48, 67, 59, 64, 71, 46, 63, 61]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Points trend
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=gameweeks, y=team_points,
            mode='lines+markers',
            name='Your Team',
            line=dict(color='#1f77b4', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=gameweeks, y=average_points,
            mode='lines+markers', 
            name='Average',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title="Points Trend vs Average",
            xaxis_title="Gameweek",
            yaxis_title="Points",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Position analysis
        positions = ['GKP', 'DEF', 'MID', 'FWD']
        avg_points_by_pos = [4.2, 15.8, 28.4, 16.1]
        
        fig = px.bar(
            x=positions, y=avg_points_by_pos,
            title="Average Points by Position",
            color=avg_points_by_pos,
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Points", "854", "+12 vs avg")
    
    with col2:
        st.metric("Overall Rank", "127,432", "⬆️ +5,248")
    
    with col3:
        st.metric("Gameweek Rank", "45,672", "⬇️ -2,108")
    
    with col4:
        st.metric("Team Value", "£101.2m", "+£1.2m")
    
    # Captain performance analysis
    st.subheader("Captain Performance Analysis")
    
    captain_data = {
        'Gameweek': gameweeks,
        'Captain': ['Haaland', 'Salah', 'Haaland', 'Kane', 'Haaland', 'Haaland', 'Son', 'Salah', 'Haaland', 'Kane', 'Salah', 'Haaland', 'Salah', 'Haaland'],
        'Points': [14, 8, 24, 4, 16, 22, 6, 18, 12, 14, 20, 2, 16, 10],
        'Captaincy Points': [28, 16, 48, 8, 32, 44, 12, 36, 24, 28, 40, 4, 32, 20]
    }
    
    captain_df = pd.DataFrame(captain_data)
    
    fig = px.bar(
        captain_df, x='Gameweek', y='Captaincy Points',
        color='Captain',
        title="Captain Points by Gameweek",
        hover_data=['Points']
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Additional utility functions for the Streamlit app

def create_formation_visualization(team_data):
    """Create a visual representation of team formation"""
    
    # This would create a football pitch visualization
    # For now, return a simple formation display
    formation_map = {
        'GKP': [(400, 50)],  # Goalkeeper position
        'DEF': [(200, 200), (400, 200), (600, 200)],  # Defender positions
        'MID': [(150, 350), (300, 350), (500, 350), (650, 350)],  # Midfielder positions  
        'FWD': [(300, 500), (500, 500), (400, 520)]  # Forward positions
    }
    
    return formation_map

def calculate_team_stats(players_data):
    """Calculate various team statistics"""
    
    total_goals = sum(p.get('goals_scored', 0) for p in players_data)
    total_assists = sum(p.get('assists', 0) for p in players_data)
    total_clean_sheets = sum(p.get('clean_sheets', 0) for p in players_data if p.get('position') in ['GKP', 'DEF'])
    avg_age = sum(p.get('age', 25) for p in players_data) / len(players_data)
    
    return {
        'total_goals': total_goals,
        'total_assists': total_assists, 
        'total_clean_sheets': total_clean_sheets,
        'avg_age': avg_age
    }

if __name__ == "__main__":
    main()