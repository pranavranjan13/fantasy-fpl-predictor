from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from typing import Optional

app = FastAPI(title="Fantasy Football Prediction API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
data_fetcher = FPLDataFetcher()
predictor = PlayerPerformancePredictor()
knowledge_base = None  # Initialize with Euri AI API key

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    global knowledge_base
    # Initialize RAG system with Euri AI
    euri_api_key = os.getenv("EURI_API_KEY")
    if euri_api_key:
        knowledge_base = FPLKnowledgeBase(
            euri_api_key=euri_api_key,
            model="gpt-4.1-nano"  # Fast and efficient model for strategic advice
        )
        knowledge_base.create_knowledge_base([
            FPL_GENERAL_STRATEGY_CONTEXT,
            FPL_CHIP_STRATEGY_CONTEXT, 
            FPL_CAPTAIN_STRATEGY_CONTEXT,
            FPL_TRANSFER_STRATEGY_CONTEXT
        ])
        print("Knowledge base initialized with Euri AI")
    else:
        print("Warning: EURI_API_KEY not found. RAG system will not be available.")

@app.get("/api/players")
async def get_players():
    """Get all current players"""
    try:
        bootstrap_data = data_fetcher.fetch_bootstrap_data()
        players = data_fetcher.process_bootstrap_to_players(bootstrap_data)
        return {"players": [player.dict() for player in players]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict-team")
async def predict_team(request: PredictionRequest):
    """Generate optimal team prediction"""
    try:
        # Fetch current player data
        bootstrap_data = data_fetcher.fetch_bootstrap_data()
        players = data_fetcher.process_bootstrap_to_players(bootstrap_data)
        
        # Train ML models (in production, this would be cached)
        player_df = predictor.prepare_features(players)
        predictor.train_position_models(player_df)
        
        # Get predictions
        predictions = predictor.predict_next_gameweek_points(players)
        
        # Optimize team
        constraints = request.constraints or TeamConstraints()
        optimizer = TeamOptimizer(constraints)
        optimal_team = optimizer.optimize_team(players, predictions)
        
        return optimal_team.dict()
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chip-recommendation")
async def get_chip_recommendation(gameweek: int, available_chips: List[str] = None):
    """Get chip usage recommendation"""
    try:
        fixtures = data_fetcher.fetch_fixtures()
        
        if knowledge_base and available_chips:
            recommendation = knowledge_base.analyze_chip_strategy(
                gameweek, fixtures, available_chips
            )
            return recommendation.dict()
        else:
            # Fallback recommendation without RAG
            return ChipRecommendation(
                chip=ChipType.WILDCARD,
                recommended_gameweek=gameweek + 1,
                confidence=0.6,
                reasoning="Consider using wildcard during favorable fixture runs",
                expected_benefit=10.0
            ).dict()
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/strategic-advice")
async def get_strategic_advice(query: str):
    """Get strategic advice using RAG"""
    try:
        if knowledge_base:
            advice = knowledge_base.get_strategic_advice(query)
            return {"advice": advice}
        else:
            return {"advice": "Strategic advice system not available. Please set EURI_API_KEY."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/player/{player_id}")
async def get_player_details(player_id: int):
    """Get detailed player information"""
    try:
        player_data = data_fetcher.fetch_player_data(player_id)
        return player_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
