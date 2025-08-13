import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Euri AI Configuration
    EURI_API_KEY = os.getenv("EURI_API_KEY")
    EURI_MODEL = os.getenv("EURI_MODEL", "gemini-2.5-pro")
    
    # Application Configuration
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    HOST = os.getenv("HOST", "0.0.0.0")
    FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", "8000"))
    STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))
    
    # FPL API
    FPL_BASE_URL = os.getenv("FPL_BASE_URL", "https://fantasy.premierleague.com/api")
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "app.log")

settings = Settings()