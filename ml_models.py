import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from euriai.langchain import create_chat_model, create_embeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import requests
import json
from typing import List, Dict, Tuple, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

class FPLMLPredictor:
    """Advanced ML-based FPL predictions with multiple models"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        self.is_trained = False
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML models"""
        features_df = df.copy()
        
        # Create advanced features
        features_df['points_per_game_ratio'] = features_df['points_per_game'] / (features_df['points_per_game'].mean() + 1e-6)
        features_df['form_trend'] = pd.to_numeric(features_df['form'], errors='coerce')
        features_df['value_efficiency'] = features_df['total_points'] / (features_df['now_cost'] / 10)
        features_df['minutes_consistency'] = features_df['minutes'] / max(features_df['minutes'].max(), 1)
        features_df['influence_factor'] = pd.to_numeric(features_df['influence'], errors='coerce') / 100
        features_df['creativity_factor'] = pd.to_numeric(features_df['creativity'], errors='coerce') / 100
        features_df['threat_factor'] = pd.to_numeric(features_df['threat'], errors='coerce') / 100
        features_df['ict_index'] = pd.to_numeric(features_df['ict_index'], errors='coerce')
        
        # Position encoding
        if not hasattr(self, 'position_encoder'):
            self.position_encoder = LabelEncoder()
            features_df['position_encoded'] = self.position_encoder.fit_transform(features_df['position'])
        else:
            features_df['position_encoded'] = self.position_encoder.transform(features_df['position'])
        
        # Team encoding
        if not hasattr(self, 'team_encoder'):
            self.team_encoder = LabelEncoder()
            features_df['team_encoded'] = self.team_encoder.fit_transform(features_df['team_name'])
        else:
            features_df['team_encoded'] = self.team_encoder.transform(features_df['team_name'])
        
        # Select feature columns
        self.feature_columns = [
            'points_per_game', 'form_trend', 'value_efficiency', 'minutes_consistency',
            'influence_factor', 'creativity_factor', 'threat_factor', 'ict_index',
            'position_encoded', 'team_encoded', 'now_cost', 'selected_by_percent'
        ]
        
        # Handle missing values
        for col in self.feature_columns:
            if col in features_df.columns:
                features_df[col] = features_df[col].fillna(features_df[col].mean())
            else:
                features_df[col] = 0
        
        return features_df[self.feature_columns]
    
    def train_models(self, df: pd.DataFrame, target_col: str = 'total_points'):
        """Train multiple ML models"""
        features_df = self.prepare_features(df)
        target = df[target_col].fillna(df[target_col].mean())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, target, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize models
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boost': GradientBoostingRegressor(random_state=42),
            'linear_regression': LinearRegression()
        }
        
        # Train models and evaluate
        model_scores = {}
        for name, model in self.models.items():
            if name == 'linear_regression':
                model.fit(X_train_scaled, y_train)
                predictions = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            model_scores[name] = {'mae': mae, 'r2': r2}
            
            print(f"{name} - MAE: {mae:.2f}, R²: {r2:.2f}")
        
        self.is_trained = True
        return model_scores
    
    def predict_points(self, df: pd.DataFrame, model_name: str = 'random_forest') -> np.ndarray:
        """Predict points using trained models"""
        if not self.is_trained:
            raise ValueError("Models not trained yet. Call train_models() first.")
        
        features_df = self.prepare_features(df)
        
        if model_name == 'linear_regression':
            features_scaled = self.scaler.transform(features_df)
            return self.models[model_name].predict(features_scaled)
        else:
            return self.models[model_name].predict(features_df)
    
    def ensemble_predict(self, df: pd.DataFrame) -> np.ndarray:
        """Ensemble prediction using all models"""
        if not self.is_trained:
            raise ValueError("Models not trained yet. Call train_models() first.")
        
        predictions = {}
        weights = {'random_forest': 0.4, 'gradient_boost': 0.4, 'linear_regression': 0.2}
        
        for name in self.models.keys():
            predictions[name] = self.predict_points(df, name)
        
        # Weighted ensemble
        ensemble_pred = np.zeros(len(df))
        for name, weight in weights.items():
            ensemble_pred += weight * predictions[name]
        
        return ensemble_pred

class FPLRAGSystem:
    """RAG system for FPL knowledge and historical data using EuriAI"""
    
    def __init__(self, euriai_api_key: Optional[str] = None):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.PersistentClient(path="./fpl_knowledge_base")
        self.collection = None
        self.euriai_chat_model = None
        self.euriai_embeddings = None
        
        if euriai_api_key:
            try:
                self.euriai_chat_model = create_chat_model(api_key=euriai_api_key)
                self.euriai_embeddings = create_embeddings(api_key=euriai_api_key)
                print("✓ EuriAI integration initialized")
            except Exception as e:
                print(f"⚠️  EuriAI initialization failed: {e}")
                print("Falling back to local embeddings")
        
        self.initialize_knowledge_base()
    
    def initialize_knowledge_base(self):
        """Initialize the knowledge base with FPL data and strategies"""
        try:
            self.collection = self.client.get_collection("fpl_knowledge")
        except:
            # Create collection with EuriAI embeddings if available
            if self.euriai_embeddings:
                # Custom embedding function for EuriAI
                class EuriAIEmbeddingFunction:
                    def __init__(self, euriai_embeddings):
                        self.euriai_embeddings = euriai_embeddings
                    
                    def __call__(self, input_texts):
                        return self.euriai_embeddings.embed_documents(input_texts)
                
                embedding_function = EuriAIEmbeddingFunction(self.euriai_embeddings)
            else:
                # Fallback to sentence transformers
                embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="gemini-embedding-001"
                )
            
            self.collection = self.client.create_collection(
                name="fpl_knowledge",
                embedding_function=embedding_function
            )
            self._populate_knowledge_base()
    
    def _populate_knowledge_base(self):
        """Populate with FPL strategies and historical insights"""
        fpl_knowledge = [
            {
                "id": "captain_strategy",
                "text": "Captain selection should prioritize players with double gameweeks, favorable fixtures, and consistent form. Historical data shows premium forwards and midfielders from top 6 teams have highest captain returns.",
                "category": "strategy"
            },
            {
                "id": "wildcard_timing",
                "text": "Optimal wildcard usage: First wildcard during international break (GW 7-10) for team restructure. Second wildcard before double gameweeks (GW 19-25) to maximize bench boost potential.",
                "category": "chips"
            },
            {
                "id": "differential_picks",
                "text": "Differential picks (owned by <5% managers) can provide significant rank improvement. Focus on players from mid-table teams with good fixtures and underlying stats.",
                "category": "strategy"
            },
            {
                "id": "fixture_analysis",
                "text": "Fixture difficulty rating (FDR) should be combined with team's attacking/defensive strength. Consider underlying stats like xG, xA, and expected points rather than just results.",
                "category": "analysis"
            },
            {
                "id": "budget_allocation",
                "text": "Optimal budget allocation: 60-65% on outfield players, premium goalkeeper not essential. Invest in premium midfielders/forwards with high ceiling potential.",
                "category": "team_building"
            },
            {
                "id": "transfer_strategy",
                "text": "Avoid early transfers unless injury/suspension. Bank transfers for double gameweeks. Consider price changes when planning transfers 2 weeks ahead.",
                "category": "transfers"
            },
            {
                "id": "bench_strategy",
                "text": "Bench should have 3.9-4.0M defenders from teams with good fixtures. First bench player should be a midfielder/forward likely to play.",
                "category": "team_building"
            },
            {
                "id": "form_vs_fixtures",
                "text": "Short-term form (3-4 games) often more predictive than long-term averages. However, fixture swing can override form for 3-4 gameweek stretches.",
                "category": "analysis"
            }
        ]
        
        # Add documents to collection
        for doc in fpl_knowledge:
            self.collection.add(
                documents=[doc["text"]],
                metadatas=[{"category": doc["category"]}],
                ids=[doc["id"]]
            )
    
    def query_knowledge_base(self, query: str, n_results: int = 3) -> List[Dict]:
        """Query the knowledge base for relevant information"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            return [
                {
                    "text": doc,
                    "metadata": meta,
                    "distance": dist
                }
                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                )
            ]
        except Exception as e:
            print(f"Error querying knowledge base: {e}")
            return []
    
    def get_ai_recommendation(self, query: str, context_data: Dict = None) -> str:
        """Get AI-powered recommendations using RAG with EuriAI"""
        # Query knowledge base
        relevant_docs = self.query_knowledge_base(query)
        
        # Build context
        context = "FPL Knowledge Base:\n"
        for i, doc in enumerate(relevant_docs, 1):
            context += f"{i}. {doc['text']}\n"
        
        if context_data:
            context += f"\nCurrent Data: {json.dumps(context_data, indent=2)}\n"
        
        # Generate response with EuriAI (if available) or return context-based response
        if self.euriai_chat_model:
            try:
                messages = [
                    SystemMessage(content="You are an expert FPL (Fantasy Premier League) analyst with deep knowledge of player performance, team strategies, and optimal decision-making. Use the provided knowledge base and current data to give specific, actionable recommendations that will help users improve their FPL performance."),
                    HumanMessage(content=f"Query: {query}\n\nContext:\n{context}\n\nProvide specific, actionable FPL recommendations based on this information. Focus on practical advice that can be immediately implemented.")
                ]
                
                response = self.euriai_chat_model.invoke(messages)
                
                # Log usage metadata if available
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    print(f"EuriAI Usage: {response.usage_metadata}")
                
                return response.content
                
            except Exception as e:
                print(f"Error calling EuriAI API: {e}")
                print("Falling back to knowledge base results")
        
        # Fallback: return enhanced knowledge base results
        response = f"Based on FPL strategies and best practices:\n\n"
        for i, doc in enumerate(relevant_docs, 1):
            response += f"{i}. {doc['text']}\n\n"
        
        # Add contextual analysis if data provided
        if context_data:
            response += f"Considering your current situation: "
            if 'period' in context_data:
                response += f"For {context_data['period']}, focus on the strategies mentioned above. "
            if 'teams' in context_data:
                response += f"Pay special attention to players from {', '.join(context_data['teams'][:3])}. "
        
        return response

class AdvancedFPLOptimizer:
    """Advanced optimization using genetic algorithm and constraint programming"""
    
    def __init__(self, players_df: pd.DataFrame):
        self.players_df = players_df
        self.ml_predictor = FPLMLPredictor()
        self.rag_system = FPLRAGSystem()
        
    def genetic_team_optimization(self, 
                                 budget: float = 100.0, 
                                 population_size: int = 100, 
                                 generations: int = 50) -> Dict:
        """Optimize team using genetic algorithm"""
        
        # Train ML model first
        try:
            self.ml_predictor.train_models(self.players_df)
            predicted_points = self.ml_predictor.ensemble_predict(self.players_df)
            players_with_predictions = self.players_df.copy()
            players_with_predictions['ml_predicted_points'] = predicted_points
        except Exception as e:
            print(f"ML training failed: {e}, using fallback predictions")
            players_with_predictions = self.players_df.copy()
            players_with_predictions['ml_predicted_points'] = players_with_predictions['total_points']
        
        def create_individual():
            """Create a random valid team"""
            team = []
            positions = {'Goalkeeper': 2, 'Defender': 5, 'Midfielder': 5, 'Forward': 3}
            
            for pos, count in positions.items():
                pos_players = players_with_predictions[
                    players_with_predictions['position'] == pos
                ].sample(n=min(count * 3, len(players_with_predictions[players_with_predictions['position'] == pos])))
                
                selected = pos_players.nlargest(count, 'ml_predicted_points')
                team.extend(selected.index.tolist())
            
            return team[:15]  # Ensure exactly 15 players
        
        def calculate_fitness(individual):
            """Calculate fitness score for a team"""
            if len(individual) != 15:
                return -1000
            
            team_data = players_with_predictions.loc[individual]
            
            # Check constraints
            total_cost = (team_data['now_cost'] / 10).sum()
            if total_cost > budget:
                return -1000
            
            # Check position constraints
            positions = team_data['position'].value_counts()
            if (positions.get('Goalkeeper', 0) != 2 or
                positions.get('Defender', 0) != 5 or
                positions.get('Midfielder', 0) != 5 or
                positions.get('Forward', 0) != 3):
                return -500
            
            # Check team constraints (max 3 players per team)
            team_counts = team_data['team'].value_counts()
            if team_counts.max() > 3:
                return -300
            
            # Calculate fitness
            predicted_points = team_data['ml_predicted_points'].sum()
            budget_efficiency = predicted_points / total_cost
            
            return predicted_points + (budget_efficiency * 10)
        
        def crossover(parent1, parent2):
            """Create offspring from two parents"""
            # Simple crossover: take random players from each parent
            child = []
            positions = {'Goalkeeper': 2, 'Defender': 5, 'Midfielder': 5, 'Forward': 3}
            
            for pos, count in positions.items():
                p1_pos = [p for p in parent1 if players_with_predictions.loc[p, 'position'] == pos]
                p2_pos = [p for p in parent2 if players_with_predictions.loc[p, 'position'] == pos]
                
                # Randomly select from both parents
                combined = list(set(p1_pos + p2_pos))
                np.random.shuffle(combined)
                child.extend(combined[:count])
            
            return child
        
        def mutate(individual, mutation_rate=0.1):
            """Mutate an individual"""
            if np.random.random() < mutation_rate:
                # Replace random player with similar one
                pos_idx = np.random.randint(0, len(individual))
                player_pos = players_with_predictions.loc[individual[pos_idx], 'position']
                
                # Find replacement from same position
                pos_players = players_with_predictions[
                    players_with_predictions['position'] == player_pos
                ].index.tolist()
                
                replacement = np.random.choice([p for p in pos_players if p not in individual])
                individual[pos_idx] = replacement
            
            return individual
        
        # Initialize population
        population = [create_individual() for _ in range(population_size)]
        
        best_fitness = -float('inf')
        best_individual = None
        
        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = [calculate_fitness(ind) for ind in population]
            
            # Track best individual
            gen_best_idx = np.argmax(fitness_scores)
            if fitness_scores[gen_best_idx] > best_fitness:
                best_fitness = fitness_scores[gen_best_idx]
                best_individual = population[gen_best_idx].copy()
            
            # Selection (tournament selection)
            new_population = []
            for _ in range(population_size):
                tournament = np.random.choice(len(population), size=3, replace=False)
                winner = tournament[np.argmax([fitness_scores[i] for i in tournament])]
                new_population.append(population[winner].copy())
            
            # Crossover and mutation
            for i in range(0, population_size - 1, 2):
                if np.random.random() < 0.8:  # Crossover probability
                    child1 = crossover(new_population[i], new_population[i + 1])
                    child2 = crossover(new_population[i + 1], new_population[i])
                    new_population[i] = mutate(child1)
                    new_population[i + 1] = mutate(child2)
            
            population = new_population
        
        # Return best team
        if best_individual:
            best_team_data = players_with_predictions.loc[best_individual]
            
            team_result = []
            for _, player in best_team_data.iterrows():
                team_result.append({
                    'name': player['web_name'],
                    'position': player['position'],
                    'team': player['team_name'],
                    'cost': player['now_cost'] / 10,
                    'predicted_points': player['ml_predicted_points'],
                    'form': player.get('form', 0)
                })
            
            return {
                'team': team_result,
                'total_cost': (best_team_data['now_cost'] / 10).sum(),
                'predicted_total_points': best_team_data['ml_predicted_points'].sum(),
                'fitness_score': best_fitness
            }
        
        return {'error': 'Optimization failed'}
    
    def get_transfer_recommendations(self, current_team: List[str], budget: float = 2.0) -> Dict:
        """Get AI-powered transfer recommendations"""
        
        # Query RAG system for transfer advice
        query = f"What are the best transfer strategies for gameweek planning with £{budget}M budget?"
        ai_advice = self.rag_system.get_ai_recommendation(query)
        
        # Analyze current team performance
        current_players = self.players_df[self.players_df['web_name'].isin(current_team)]
        
        if self.ml_predictor.is_trained:
            current_predictions = self.ml_predictor.ensemble_predict(current_players)
        else:
            current_predictions = current_players['total_points'].values
        
        # Find underperforming players
        current_players_analysis = current_players.copy()
        current_players_analysis['predicted_points'] = current_predictions
        current_players_analysis['performance_score'] = (
            current_players_analysis['predicted_points'] / 
            (current_players_analysis['now_cost'] / 10)
        )
        
        # Get transfer targets
        available_players = self.players_df[~self.players_df['web_name'].isin(current_team)]
        
        if self.ml_predictor.is_trained:
            available_predictions = self.ml_predictor.ensemble_predict(available_players)
        else:
            available_predictions = available_players['total_points'].values
        
        available_players_analysis = available_players.copy()
        available_players_analysis['predicted_points'] = available_predictions
        available_players_analysis['performance_score'] = (
            available_players_analysis['predicted_points'] / 
            (available_players_analysis['now_cost'] / 10)
        )
        
        # Generate recommendations
        recommendations = []
        
        for _, underperformer in current_players_analysis.nsmallest(3, 'performance_score').iterrows():
            # Find replacements in same position within budget
            position = underperformer['position']
            max_cost = (underperformer['now_cost'] / 10) + budget
            
            replacements = available_players_analysis[
                (available_players_analysis['position'] == position) &
                (available_players_analysis['now_cost'] / 10 <= max_cost)
            ].nlargest(3, 'performance_score')
            
            for _, replacement in replacements.iterrows():
                cost_diff = (replacement['now_cost'] - underperformer['now_cost']) / 10
                if cost_diff <= budget:
                    recommendations.append({
                        'out': underperformer['web_name'],
                        'in': replacement['web_name'],
                        'cost_change': cost_diff,
                        'points_improvement': replacement['predicted_points'] - underperformer['predicted_points'],
                        'reasoning': f"Better predicted performance and value"
                    })
        
        return {
            'ai_advice': ai_advice,
            'transfer_suggestions': sorted(recommendations, key=lambda x: x['points_improvement'], reverse=True)[:5]
        }

class FPLSeasonPlanner:
    """Long-term season planning with fixture analysis"""
    
    def __init__(self, players_df: pd.DataFrame):
        self.players_df = players_df
        self.rag_system = FPLRAGSystem()
    
    def analyze_fixture_swings(self, team_fixtures: Dict) -> Dict:
        """Analyze fixture difficulty swings for planning"""
        # This would typically fetch fixture data from FPL API
        # For demo purposes, we'll simulate fixture analysis
        
        fixture_periods = {
            'easy_run_gw8_12': ['Arsenal', 'Manchester City', 'Liverpool'],
            'tough_run_gw15_19': ['Sheffield United', 'Burnley', 'Luton Town'],
            'dgw_candidates_gw26': ['Manchester City', 'Arsenal', 'Liverpool'],
            'blank_gw_risks': ['FA Cup teams', 'European competition teams']
        }
        
        recommendations = {}
        for period, teams in fixture_periods.items():
            query = f"How should I plan transfers for {period} considering fixture difficulty?"
            ai_advice = self.rag_system.get_ai_recommendation(
                query, 
                {'period': period, 'teams': teams}
            )
            recommendations[period] = ai_advice
        
        return recommendations
    
    def create_season_roadmap(self) -> Dict:
        """Create comprehensive season planning roadmap"""
        
        roadmap = {
            'GW1-7': {
                'strategy': 'Initial team assessment and minor tweaks',
                'key_actions': ['Monitor new signings', 'Assess early form', 'Bank transfers'],
                'chip_usage': 'Save all chips'
            },
            'GW8-15': {
                'strategy': 'First major team restructure',
                'key_actions': ['Use first wildcard', 'Target fixture swings', 'Build team value'],
                'chip_usage': 'Wildcard 1'
            },
            'GW16-25': {
                'strategy': 'Prepare for double gameweeks',
                'key_actions': ['Bank transfers', 'Target DGW players', 'Plan chip usage'],
                'chip_usage': 'Save for DGWs'
            },
            'GW26-30': {
                'strategy': 'Maximize double gameweeks',
                'key_actions': ['Triple Captain premium DGW player', 'Bench Boost with DGW bench'],
                'chip_usage': 'Triple Captain, Bench Boost'
            },
            'GW31-38': {
                'strategy': 'Final push optimization',
                'key_actions': ['Target form players', 'Consider differentials', 'Free hit blank GW'],
                'chip_usage': 'Free Hit, Wildcard 2'
            }
        }
        
        # Get AI insights for each phase
        for phase, details in roadmap.items():
            query = f"What are the key strategies for {phase} in FPL season planning?"
            ai_insights = self.rag_system.get_ai_recommendation(query, details)
            roadmap[phase]['ai_insights'] = ai_insights
        
        return roadmap

# Usage example and testing functions
def test_advanced_features():
    """Test the advanced ML and RAG features"""
    
    # Sample data for testing
    sample_data = pd.DataFrame({
        'web_name': ['Player A', 'Player B', 'Player C'],
        'position': ['Midfielder', 'Forward', 'Defender'],
        'team_name': ['Arsenal', 'Liverpool', 'Chelsea'],
        'total_points': [150, 120, 80],
        'now_cost': [100, 90, 50],
        'form': ['5.0', '4.0', '3.0'],
        'points_per_game': [6.5, 5.5, 3.0],
        'minutes': [2500, 2200, 1800],
        'influence': ['800', '600', '400'],
        'creativity': ['700', '500', '300'],
        'threat': ['900', '700', '200'],
        'ict_index': ['24.0', '18.0', '9.0'],
        'selected_by_percent': [25.5, 15.2, 8.1],
        'team': [1, 2, 3]
    })
    
    # Test ML predictor
    print("Testing ML Predictor...")
    ml_predictor = FPLMLPredictor()
    try:
        scores = ml_predictor.train_models(sample_data)
        predictions = ml_predictor.ensemble_predict(sample_data)
        print(f"Predictions: {predictions}")
        print("ML Predictor: ✓ Working")
    except Exception as e:
        print(f"ML Predictor Error: {e}")
    
    # Test RAG system
    print("\nTesting RAG System...")
    rag_system = FPLRAGSystem()
    try:
        recommendation = rag_system.get_ai_recommendation("Should I use my wildcard now?")
        print(f"RAG Recommendation: {recommendation[:100]}...")
        print("RAG System: ✓ Working")
    except Exception as e:
        print(f"RAG System Error: {e}")
    
    print("\nAdvanced features test completed!")

if __name__ == "__main__":
    test_advanced_features()
