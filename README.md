# Fantasy Football Prediction Application

A comprehensive AI-powered fantasy football application that predicts optimal team selections, provides strategic advice, and recommends chip usage timing.

## Features

- **Team Optimization**: Uses linear programming to select optimal 15-player squads
- **ML Predictions**: Ensemble models (Random Forest, XGBoost, Gradient Boosting) for player performance prediction
- **RAG System**: Retrieval-Augmented Generation for strategic advice and chip recommendations
- **Real-time Data**: Fetches live data from Fantasy Premier League API
- **Interactive Dashboard**: Streamlit-based frontend with visualizations and analytics

## Architecture

- **Backend**: FastAPI with async endpoints
- **Frontend**: Streamlit with interactive visualizations
- **ML Engine**: Scikit-learn, XGBoost for predictions
- **RAG System**: LangChain + OpenAI + ChromaDB
- **Optimization**: PuLP for linear programming
- **Data Source**: Official FPL API

## Installation

1. **Clone the repository**

```bash
git clone <repository-url>
cd fantasy-football-predictor
```
