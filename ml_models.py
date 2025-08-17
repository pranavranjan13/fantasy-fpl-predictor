import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import requests
import json
from typing import List, Dict, Tuple, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

class EnhancedFPLMLPredictor:
    """Enhanced ML predictor with historical data integration"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        self.is_trained = False
        
    def prepare_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare enhanced features including historical performance"""
        features_df = df.copy()
        
        # Enhanced feature engineering
        features_df['points_per_game_ratio'] = features_df['points_per_game'] / (features_df['points_per_game'].mean() + 1e-6)
        features_df['form_trend'] = pd.to_numeric(features_df['form'], errors='coerce')
        features_df['value_efficiency'] = features_df['total_points'] / (features_df['now_cost'] / 10)
        features_df['minutes_consistency'] = features_df['minutes'] / max(features_df['minutes'].max(), 1)
        
        # ICT index components
        features_df['influence_factor'] = pd.to_numeric(features_df['influence'], errors='coerce') / 100
        features_df['creativity_factor'] = pd.to_numeric(features_df['creativity'], errors='coerce') / 100
        features_df['threat_factor'] = pd.to_numeric(features_df['threat'], errors='coerce') / 100
        features_df['ict_index'] = pd.to_numeric(features_df['ict_index'], errors='coerce')
        
        # New enhanced features
        features_df['goals_per_game'] = features_df.get('goals_scored', 0) / max(features_df.get('games_played', 1), 1)
        features_df['assists_per_game'] = features_df.get('assists', 0) / max(features_df.get('games_played', 1), 1)
        features_df['bonus_points_ratio'] = features_df.get('bonus', 0) / max(features_df['total_points'], 1)
        
        # Historical performance features
        if 'historical_score' in features_df.columns:
            features_df['historical_vs_current'] = features_df['historical_score'] / max(features_df['total_points'], 1)
        else:
            features_df['historical_vs_current'] = 1.0
        
        # Consistency metrics
        if 'consistency_score' in features_df.columns:
            features_df['consistency_factor'] = features_df['consistency_score'] / 10
        else:
            features_df['consistency_factor'] = 0.5
        
        # Starting XI probability
        if 'starting_xi_probability' in features_df.columns:
            features_df['starting_reliability'] = features_df['starting_xi_probability']
        else:
            features_df['starting_reliability'] = 0.5
        
        # Position and team encoding
        if not hasattr(self, 'position_encoder'):
            self.position_encoder = LabelEncoder()
            features_df['position_encoded'] = self.position_encoder.fit_transform(features_df['position'])
        else:
            try:
                features_df['position_encoded'] = self.position_encoder.transform(features_df['position'])
            except ValueError:
                # Handle unseen positions
                features_df['position_encoded'] = 0
        
        if not hasattr(self, 'team_encoder'):
            self.team_encoder = LabelEncoder()
            features_df['team_encoded'] = self.team_encoder.fit_transform(features_df['team_name'])
        else:
            try:
                features_df['team_encoded'] = self.team_encoder.transform(features_df['team_name'])
            except ValueError:
                # Handle unseen teams
                features_df['team_encoded'] = 0
        
        # Final feature selection
        self.feature_columns = [
            'points_per_game', 'form_trend', 'value_efficiency', 'minutes_consistency',
            'influence_factor', 'creativity_factor', 'threat_factor', 'ict_index',
            'goals_per_game', 'assists_per_game', 'bonus_points_ratio',
            'historical_vs_current', 'consistency_factor', 'starting_reliability',
            'position_encoded', 'team_encoded', 'now_cost', 'selected_by_percent'
        ]
        
        # Handle missing values
        for col in self.feature_columns:
            if col in features_df.columns:
                features_df[col] = features_df[col].fillna(features_df[col].mean())
            else:
                features_df[col] = 0.5 if 'factor' in col or 'reliability' in col else 0
        
        return features_df[self.feature_columns]
    
    def train_enhanced_models(self, df: pd.DataFrame, target_col: str = 'total_points'):
        """Train enhanced ML models with cross-validation"""
        features_df = self.prepare_enhanced_features(df)
        target = df[target_col].fillna(df[target_col].mean())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, target, test_size=0.2, random_state=42, stratify=df['position']
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Enhanced model ensemble
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=150, 
                max_depth=12, 
                min_samples_split=5,
                random_state=42
            ),
            'gradient_boost': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'linear_regression': LinearRegression()
        }
        
        # Train and evaluate models
        model_scores = {}
        cv_scores = {}
        
        for name, model in self.models.items():
            if name == 'linear_regression':
                model.fit(X_train_scaled, y_train)
                predictions = model.predict(X_test_scaled)
                # Cross-validation on scaled data
                cv_score = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_absolute_error')
            else:
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                # Cross-validation on original data
                cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
            
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            cv_mae = -cv_score.mean()
            
            model_scores[name] = {'mae': mae, 'r2': r2, 'cv_mae': cv_mae}
            cv_scores[name] = cv_score
            
            print(f"{name} - MAE: {mae:.2f}, R²: {r2:.2f}, CV MAE: {cv_mae:.2f}")
        
        self.is_trained = True
        return model_scores, cv_scores
    
    def predict_enhanced_points(self, df: pd.DataFrame, model_name: str = 'ensemble') -> np.ndarray:
        """Enhanced prediction with ensemble option"""
        if not self.is_trained:
            raise ValueError("Models not trained yet. Call train_enhanced_models() first.")
        
        features_df = self.prepare_enhanced_features(df)
        
        if model_name == 'ensemble':
            return self.ensemble_predict_enhanced(df)
        elif model_name == 'linear_regression':
            features_scaled = self.scaler.transform(features_df)
            return self.models[model_name].predict(features_scaled)
        else:
            return self.models[model_name].predict(features_df)
    
    def ensemble_predict_enhanced(self, df: pd.DataFrame) -> np.ndarray:
        """Enhanced ensemble prediction with position-specific weights"""
        if not self.is_trained:
            raise ValueError("Models not trained yet. Call train_enhanced_models() first.")
        
        predictions = {}
        
        # Position-specific model weights
        position_weights = {
            'Goalkeeper': {'random_forest': 0.5, 'gradient_boost': 0.3, 'linear_regression': 0.2},
            'Defender': {'random_forest': 0.4, 'gradient_boost': 0.4, 'linear_regression': 0.2},
            'Midfielder': {'random_forest': 0.35, 'gradient_boost': 0.45, 'linear_regression': 0.2},
            'Forward': {'random_forest': 0.3, 'gradient_boost': 0.5, 'linear_regression': 0.2}
        }
        
        for name in self.models.keys():
            predictions[name] = self.predict_enhanced_points(df, name)
        
        # Position-aware ensemble
        ensemble_pred = np.zeros(len(df))
        
        for i, (_, player) in enumerate(df.iterrows()):
            position = player.get('position', 'Midfielder')
            weights = position_weights.get(position, position_weights['Midfielder'])
            
            for model_name, weight in weights.items():
                ensemble_pred[i] += weight * predictions[model_name][i]
        
        return ensemble_pred

class EnhancedFPLChatBot:
    """Enhanced chatbot with EuriAI integration and comprehensive FPL knowledge"""
    
    def __init__(self, euriai_api_key: str = None):
        self.euriai_api_key = euriai_api_key
        self.chat_history = []
        self.conversation_context = {}
        
        # Enhanced FPL knowledge base
        self.fpl_knowledge = {
            "captain_strategy": {
                "content": "Captain selection priorities: 1) Double gameweek players 2) Premium players vs weak opponents 3) Form over fixtures for short-term 4) Avoid rotation risks 5) Consider differential captains for rank climbing",
                "examples": ["Haaland vs Brighton (H)", "Salah in double gameweek", "Son vs Sheffield United"]
            },
            "wildcard_timing": {
                "content": "Optimal wildcard usage: WC1 during international break (GW7-10) for team restructure after initial assessment. WC2 before double gameweeks (GW19-25) to maximize bench boost potential.",
                "examples": ["October international break", "January prep for DGWs"]
            },
            "transfer_strategy": {
                "content": "Transfer principles: Bank transfers for DGWs, avoid early moves unless injury/suspension, consider price changes 2 weeks ahead, target fixture swings over 3-4 GWs, don't chase last week's points",
                "examples": ["Banking for BGW/DGW", "Fixture swing transfers", "Price rise planning"]
            },
            "formation_tactics": {
                "content": "Formation selection based on team structure: 3-5-2 for premium midfielder strategy, 3-4-3 for attacking approach, 4-4-2 most balanced, 5-3-2 when owning premium defenders, 4-5-1 for ultra-premium mid strategy",
                "examples": ["3-5-2 with Salah+KDB+Son", "3-4-3 chase formation", "5-3-2 with premium defenders"]
            },
            "chip_optimization": {
                "content": "Chip timing: TC on premium DGW players, BB when entire bench has fixtures, FH during blank gameweeks, WC during international breaks or before fixture swings",
                "examples": ["TC Haaland DGW26", "BB with 4 DGW players", "FH blank GW29"]
            },
            "budget_allocation": {
                "content": "Optimal spending: 60-65% on outfield players, max 4.5M on GK, invest in 2-3 premiums (8M+), balance with 4.5M defenders, leave 0.5-1M ITB for flexibility",
                "examples": ["15M on 3 premiums", "4.5M defender rotation", "4.5M GK strategy"]
            },
            "differential_strategy": {
                "content": "Differential picks for rank improvement: Target <5% ownership, focus on fixture swings, avoid punts in defense, consider mid-table team assets, time differentials with DGWs",
                "examples": ["Mitoma fixture swing", "Watkins vs easy fixtures", "Newcastle assets"]
            },
            "fixture_analysis": {
                "content": "Fixture assessment: Use FDR + team stats, consider home/away splits, factor in European competition, analyze underlying stats (xG/xGA), weight recent form vs fixtures",
                "examples": ["Arsenal home fortress", "City away struggles", "Europa League rotation"]
            }
        }
    
    def get_enhanced_response(self, user_message: str, context_data: Dict = None) -> str:
        """Get enhanced AI response with context awareness"""
        
        # Update conversation context
        if context_data:
            self.conversation_context.update(context_data)
        
        # Add to chat history
        self.chat_history.append({"role": "user", "content": user_message})
        
        if self.euriai_api_key:
            try:
                response = self._get_euriai_enhanced_response(user_message, context_data)
                self.chat_history.append({"role": "assistant", "content": response})
                return response
            except Exception as e:
                print(f"EuriAI error: {e}")
                return self._get_enhanced_fallback_response(user_message, context_data)
        else:
            return self._get_enhanced_fallback_response(user_message, context_data)
    
    def _get_euriai_enhanced_response(self, message: str, context_data: Dict = None) -> str:
        """Enhanced EuriAI response with better context"""
        
        # Build comprehensive system prompt
        system_prompt = """You are an elite FPL (Fantasy Premier League) expert with deep analytical knowledge of:

CORE EXPERTISE:
- Player performance analysis and statistical modeling
- Transfer strategy and market timing
- Captaincy selection and risk management
- Formation optimization and tactical flexibility
- Chip usage strategy and timing
- Budget management and team value growth
- Fixture analysis and difficulty assessment
- Differential strategy for rank improvement

ANALYSIS APPROACH:
- Use data-driven insights from current season performance
- Consider underlying statistics (xG, xA, xGI)
- Factor in fixture difficulty and team strength
- Analyze rotation risks and injury concerns
- Consider price changes and market trends
- Account for gameweek planning and chip usage

RESPONSE STYLE:
- Provide specific, actionable recommendations
- Use concrete examples and player names when relevant
- Explain the reasoning behind suggestions
- Consider both template and differential strategies
- Address risk vs reward considerations
- Include timing advice for optimal decisions

Always aim to improve the user's FPL performance through strategic, well-reasoned advice."""
        
        # Build context-aware message
        context_message = ""
        if context_data:
            context_parts = []
            if 'gameweek' in context_data:
                context_parts.append(f"Current Gameweek: {context_data['gameweek']}")
            if 'current_rank' in context_data:
                context_parts.append(f"User's Rank: {context_data['current_rank']:,}")
            if 'budget' in context_data:
                context_parts.append(f"Available Budget: £{context_data['budget']}M")
            
            if context_parts:
                context_message = f"\nCURRENT CONTEXT: {' | '.join(context_parts)}\n"
        
        # Include relevant FPL knowledge
        relevant_knowledge = self._get_relevant_knowledge(message)
        knowledge_context = ""
        if relevant_knowledge:
            knowledge_context = f"\nRELEVANT FPL KNOWLEDGE:\n{relevant_knowledge}\n"
        
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add recent chat history for context
        recent_history = self.chat_history[-4:] if len(self.chat_history) > 4 else self.chat_history
        messages.extend(recent_history)
        
        # Add current message with context
        full_message = f"{context_message}{knowledge_context}USER QUESTION: {message}"
        messages.append({"role": "user", "content": full_message})
        
        # Make EuriAI API call
        response = requests.post(
            "https://api.euron.one/api/v1/euri/chat/completions",
            headers={
                "Content-Type": "application/json",
            },
            json={
                "model": "gemini-2.5-pro",
                "max_tokens": 800,
                "messages": messages,
                "temperature": 0.5
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            return data['content'][0]['text']
        else:
            raise Exception(f"API error: {response.status_code} - {response.text}")
    
    def _get_relevant_knowledge(self, message: str) -> str:
        """Extract relevant knowledge based on message content"""
        message_lower = message.lower()
        relevant_sections = []
        
        # Map keywords to knowledge sections
        keyword_mapping = {
            ['captain', 'c', 'armband']: 'captain_strategy',
            ['transfer', 'buy', 'sell', 'in', 'out']: 'transfer_strategy',
            ['formation', 'starting', 'xi', '11', 'lineup']: 'formation_tactics',
            ['chip', 'wildcard', 'bench boost', 'triple', 'free hit']: 'chip_optimization',
            ['budget', 'money', 'price', 'value', 'cost']: 'budget_allocation',
            ['differential', 'punt', 'unique', 'rank']: 'differential_strategy',
            ['fixture', 'fdr', 'difficulty', 'easy', 'tough']: 'fixture_analysis'
        }
        
        for keywords, section in keyword_mapping.items():
            if any(keyword in message_lower for keyword in keywords):
                knowledge = self.fpl_knowledge[section]
                relevant_sections.append(f"**{section.replace('_', ' ').title()}**: {knowledge['content']}")
        
        return '\n\n'.join(relevant_sections[:2])  # Limit to 2 most relevant sections
    
    def _get_enhanced_fallback_response(self, message: str, context_data: Dict = None) -> str:
        """Enhanced fallback response system"""
        
        message_lower = message.lower()
        
        # Enhanced pattern matching
        if any(word in message_lower for word in ['captain', 'c', 'armband']):
            return self._get_enhanced_captain_advice(context_data)
        elif any(word in message_lower for word in ['transfer', 'buy', 'sell']):
            return self._get_enhanced_transfer_advice(context_data)
        elif any(word in message_lower for word in ['formation', 'starting', 'xi', '11']):
            return self._get_enhanced_formation_advice(context_data)
        elif any(word in message_lower for word in ['chip', 'wildcard', 'bench boost', 'triple']):
            return self._get_enhanced_chip_advice(context_data)
        elif any(word in message_lower for word in ['budget', 'money', 'price', 'value']):
            return self._get_enhanced_budget_advice(context_data)
        elif any(word in message_lower for word in ['differential', 'punt', 'rank']):
            return self._get_differential_advice(context_data)
        else:
            return self._get_comprehensive_advice(message, context_data)
    
    def _get_enhanced_captain_advice(self, context_data):
        """Enhanced captain selection advice"""
        advice = "👑 **Enhanced Captain Selection Strategy:**\n\n"
        advice += "🎯 **Priority Order:**\n"
        advice += "1. Premium players in double gameweeks\n"
        advice += "2. Form players vs weak opponents (FDR ≤ 3)\n"
        advice += "3. Consistent performers with high minutes\n"
        advice += "4. Home advantage for attacking players\n\n"
        
        advice += "📊 **Key Metrics to Consider:**\n"
        advice += "• Expected Goals/Assists (xG/xA)\n"
        advice += "• Recent form (last 4-6 games)\n"
        advice += "• Historical performance vs opponent\n"
        advice += "• Team news and rotation risk\n\n"
        
        if context_data:
            if context_data.get('gameweek', 0) > 25:
                advice += "🏃‍♂️ **Late Season Strategy:** Consider differentials if chasing rank\n"
            elif context_data.get('gameweek', 0) < 10:
                advice += "🛡️ **Early Season:** Stick to premium template picks\n"
        
        advice += "⚠️ **Avoid:** Defenders as captain, players with rotation risk, away fixtures vs top 6"
        
        return advice
    
    def _get_enhanced_transfer_advice(self, context_data):
        """Enhanced transfer strategy advice"""
        advice = "🔄 **Enhanced Transfer Strategy:**\n\n"
        advice += "📅 **Timing Principles:**\n"
        advice += "• Bank transfers for double gameweeks\n"
        advice += "• Plan 2-3 gameweeks ahead for price changes\n"
        advice += "• Avoid knee-jerk reactions to single gameweeks\n"
        advice += "• Use international breaks for major changes\n\n"
        
        advice += "🎯 **Target Selection:**\n"
        advice += "• Players with 3-4 gameweek fixture swings\n"
        advice += "• Consistent starters (2000+ minutes)\n"
        advice += "• Value picks before price rises\n"
        advice += "• Form players with sustainable metrics\n\n"
        
        if context_data:
            gw = context_data.get('gameweek', 1)
            if 15 <= gw <= 20:
                advice += "🎪 **Christmas Period:** Avoid rotation-prone players\n"
            elif 25 <= gw <= 30:
                advice += "⚡ **DGW Season:** Target double gameweek players\n"
        
        advice += "💡 **Pro Tips:** Monitor press conferences, check injury reports, consider team value growth"
        
        return advice
    
    def _get_enhanced_formation_advice(self, context_data):
        """Enhanced formation selection advice"""
        advice = "⚡ **Enhanced Formation Strategy:**\n\n"
        advice += "🎮 **Formation Guide:**\n\n"
        
        formations = {
            "3-5-2": "Premium midfielder heavy strategy",
            "3-4-3": "High risk, high reward attacking setup", 
            "4-4-2": "Balanced approach, safest option",
            "4-3-3": "Good mix of stability and attack",
            "5-3-2": "Premium defender strategy",
            "4-5-1": "Ultra-premium midfielder approach"
        }
        
        for formation, description in formations.items():
            advice += f"**{formation}**: {description}\n"
        
        advice += "\n🧠 **Selection Criteria:**\n"
        advice += "• Match formation to your premium players\n"
        advice += "• Consider fixture difficulty by position\n"
        advice += "• Factor in rotation risks\n"
        advice += "• Adapt to gameweek strategy (attacking vs defensive)\n\n"
        
        advice += "📈 **Advanced Tips:**\n"
        advice += "• Use 3-4-3 when chasing ranks\n"
        advice += "• Switch to 5-3-2 for tough fixtures\n"
        advice += "• Consider bench strength for formation flexibility"
        
        return advice
    
    def _get_enhanced_chip_advice(self, context_data):
        """Enhanced chip usage strategy"""
        advice = "💎 **Enhanced Chip Strategy:**\n\n"
        
        chip_strategies = {
            "Wildcard": {
                "timing": "GW7-10 (WC1), GW19-25 (WC2)",
                "strategy": "Team restructure during breaks, DGW preparation",
                "tips": "Plan 2 weeks ahead, consider price changes"
            },
            "Triple Captain": {
                "timing": "GW26, GW35 (DGWs)",
                "strategy": "Premium players in double gameweeks",
                "tips": "Avoid rotation risks, check team news"
            },
            "Bench Boost": {
                "timing": "GW26, GW34 (DGWs)",
                "strategy": "When entire bench has fixtures",
                "tips": "Build strong bench 2-3 GWs prior"
            },
            "Free Hit": {
                "timing": "GW18, GW29 (BGWs)",
                "strategy": "Blank gameweeks with few fixtures",
                "tips": "Target teams with fixtures, one-week punts"
            }
        }
        
        for chip, info in chip_strategies.items():
            advice += f"**{chip}**\n"
            advice += f"⏰ Best timing: {info['timing']}\n"
            advice += f"🎯 Strategy: {info['strategy']}\n"
            advice += f"💡 Tips: {info['tips']}\n\n"
        
        if context_data:
            gw = context_data.get('gameweek', 1)
            if 7 <= gw <= 10:
                advice += "🔥 **Current Focus:** Consider first wildcard during international break"
            elif 25 <= gw <= 30:
                advice += "🔥 **Current Focus:** Prime time for TC and BB usage"
        
        return advice
    
    def _get_enhanced_budget_advice(self, context_data):
        """Enhanced budget management advice"""
        advice = "💰 **Enhanced Budget Management:**\n\n"
        advice += "📊 **Optimal Allocation:**\n"
        advice += "• 60-65% on outfield players (60-65M)\n"
        advice += "• 2-3 premium players (24-30M total)\n"
        advice += "• 4.0-4.5M goalkeeper strategy\n"
        advice += "• 4.0-4.5M defender enablers\n"
        advice += "• Keep 0.5-1.0M ITB for flexibility\n\n"
        
        advice += "🎯 **Value Strategies:**\n"
        advice += "• Target players before price rises\n"
        advice += "• Sell before significant drops\n"
        advice += "• Monitor ownership % changes\n"
        advice += "• Consider template vs differential balance\n\n"
        
        advice += "📈 **Team Value Growth:**\n"
        advice += "• Buy early season bargains\n"
        advice += "• Hold players through good runs\n"
        advice += "• Time transfers around price change nights\n"
        advice += "• Balance growth with performance\n\n"
        
        if context_data and 'budget' in context_data:
            budget = context_data['budget']
            if budget < 1:
                advice += f"🚨 **Current Budget £{budget}M:** Focus on sideways moves only\n"
            elif budget > 2:
                advice += f"💪 **Current Budget £{budget}M:** Great flexibility for upgrades\n"
        
        return advice
    
    def _get_differential_advice(self, context_data):
        """Differential strategy advice"""
        advice = "🎯 **Differential Strategy Guide:**\n\n"
        advice += "📊 **Ownership Thresholds:**\n"
        advice += "• Template picks: >15% ownership\n"
        advice += "• Semi-differentials: 5-15% ownership\n"
        advice += "• True differentials: <5% ownership\n\n"
        
        advice += "🎮 **Strategy by Rank:**\n"
        advice += "• **Top 1M:** Mix template with 1-2 differentials\n"
        advice += "• **1M-3M:** Mostly template, safe differentials\n"
        advice += "• **3M+:** Focus on template picks first\n\n"
        
        advice += "🔍 **Differential Selection:**\n"
        advice += "• Target fixture swing players\n"
        advice += "• Focus on attacking positions\n"
        advice += "• Avoid differential defenders\n"
        advice += "• Consider mid-table team assets\n\n"
        
        advice += "⚠️ **Risk Management:**\n"
        advice += "• Limit to 2-3 differentials max\n"
        advice += "• Have template backup options\n"
        advice += "• Time differentials with favorable fixtures\n"
        
        if context_data:
            rank = context_data.get('current_rank', 1000000)
            if rank < 100000:
                advice += "\n🏆 **Your Rank Strategy:** Consider bold differentials for further climb"
            elif rank > 2000000:
                advice += "\n📈 **Your Rank Strategy:** Stick to template picks for now"
        
        return advice
    
    def _get_comprehensive_advice(self, message, context_data):
        """Comprehensive FPL guidance"""
        advice = "🏆 **Comprehensive FPL Strategy:**\n\n"
        advice += "📋 **Weekly Routine:**\n"
        advice += "1. Check injury/suspension news\n"
        advice += "2. Analyze upcoming fixtures (3-4 GWs)\n"
        advice += "3. Review underlying stats and form\n"
        advice += "4. Plan transfers and captain choice\n"
        advice += "5. Monitor price changes\n\n"
        
        advice += "📊 **Key Resources:**\n"
        advice += "• FPL official site for news/stats\n"
        advice += "• Understat for xG/xA data\n"
        advice += "• Press conferences for rotation hints\n"
        advice += "• FPL communities for insights\n\n"
        
        advice += "🎯 **Success Principles:**\n"
        advice += "• Patience over knee-jerk reactions\n"
        advice += "• Data-driven decision making\n"
        advice += "• Long-term planning with flexibility\n"
        advice += "• Risk management and diversification\n\n"
        
        if context_data:
            gw = context_data.get('gameweek', 1)
            if gw <= 10:
                advice += "🌱 **Early Season Focus:** Team assessment and initial optimizations"
            elif 11 <= gw <= 25:
                advice += "⚖️ **Mid Season Focus:** Strategic transfers and chip planning"
            else:
                advice += "🏁 **Late Season Focus:** Chip usage and final push strategy"
        
        return advice

# Integration functions for the main app
def integrate_enhanced_ml_with_app(players_df: pd.DataFrame) -> EnhancedFPLMLPredictor:
    """Initialize and train enhanced ML predictor for the app"""
    ml_predictor = EnhancedFPLMLPredictor()
    
    try:
        print("🤖 Training enhanced ML models...")
        model_scores, cv_scores = ml_predictor.train_enhanced_models(players_df)
        print("✅ ML models trained successfully!")
        return ml_predictor
    except Exception as e:
        print(f"⚠️ ML training failed: {e}")
        print("📊 Using fallback prediction methods")
        return None

def create_enhanced_chatbot(euriai_key: str = None) -> EnhancedFPLChatBot:
    """Create enhanced chatbot instance"""
    return EnhancedFPLChatBot(euriai_key)

# Testing and validation functions
def test_enhanced_features():
    """Test enhanced ML and chat features"""
    print("🧪 Testing Enhanced FPL Features...")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'web_name': ['Haaland', 'Salah', 'Alexander-Arnold', 'Alisson'],
        'position': ['Forward', 'Midfielder', 'Defender', 'Goalkeeper'],
        'team_name': ['Manchester City', 'Liverpool', 'Liverpool', 'Liverpool'],
        'total_points': [200, 180, 120, 100],
        'now_cost': [120, 130, 70, 50],
        'form': ['6.0', '5.5', '4.0', '3.5'],
        'points_per_game': [8.5, 7.2, 5.0, 4.2],
        'minutes': [2800, 2900, 2600, 2400],
        'influence': ['1000', '900', '600', '400'],
        'creativity': ['400', '800', '500', '200'],
        'threat': ['1200', '800', '300', '100'],
        'ict_index': ['26.0', '25.0', '14.0', '7.0'],
        'selected_by_percent': [45.2, 38.7, 22.1, 15.8],
        'team': [1, 2, 2, 2],
        'historical_score': [220, 200, 140, 110],
        'consistency_score': [8.5, 7.8, 6.2, 5.5],
        'starting_xi_probability': [0.95, 0.92, 0.88, 0.85],
        'goals_scored': [25, 15, 3, 0],
        'assists': [5, 12, 8, 0],
        'bonus': [20, 18, 12, 8],
        'games_played': [30, 32, 28, 25]
    })
    
    # Test enhanced ML predictor
    try:
        ml_predictor = EnhancedFPLMLPredictor()
        model_scores, cv_scores = ml_predictor.train_enhanced_models(sample_data)
        predictions = ml_predictor.predict_enhanced_points(sample_data, 'ensemble')
        print(f"✅ Enhanced ML: Predictions = {predictions[:2]}...")
        print("✅ Enhanced ML Predictor working!")
    except Exception as e:
        print(f"❌ Enhanced ML Error: {e}")
    
    # Test enhanced chatbot
    try:
        chatbot = EnhancedFPLChatBot()
        response = chatbot.get_enhanced_response("Who should I captain this week?")
        print(f"✅ Enhanced Chat: Response length = {len(response)} chars")
        print("✅ Enhanced Chatbot working!")
    except Exception as e:
        print(f"❌ Enhanced Chat Error: {e}")
    
    print("🎉 Enhanced features testing completed!")

if __name__ == "__main__":
    test_enhanced_features()