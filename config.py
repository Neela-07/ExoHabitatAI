"""
Configuration settings for ExoHabitatAI
Supports both local development and cloud deployment (Heroku/Render)
"""
import os
from pathlib import Path
from urllib.parse import urlparse

# Load environment variables from .env file (for local development)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Base directory
BASE_DIR = Path(__file__).parent

# Environment
FLASK_ENV = os.getenv("FLASK_ENV", "development")
IS_PRODUCTION = FLASK_ENV == "production"

# Data paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
RAW_DATA_DIR.mkdir(exist_ok=True)
PROCESSED_DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Parse DATABASE_URL if provided (Heroku/Render format)
DATABASE_URL = os.getenv("DATABASE_URL", "")

# Fix Heroku's postgres:// to postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

def parse_database_url(url):
    """Parse DATABASE_URL into components"""
    if not url:
        return None
    try:
        parsed = urlparse(url)
        return {
            "host": parsed.hostname,
            "port": parsed.port or 5432,
            "database": parsed.path.lstrip("/"),
            "user": parsed.username,
            "password": parsed.password
        }
    except Exception:
        return None

# Database configuration
DATABASE_CONFIG = {
    "type": "postgresql" if DATABASE_URL else os.getenv("DB_TYPE", "csv"),
    "url": DATABASE_URL,  # Full connection URL for SQLAlchemy
    "postgresql": parse_database_url(DATABASE_URL) or {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", 5432)),
        "database": os.getenv("DB_NAME", "exohabitat"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", "")
    },
    "csv": {
        "raw_file": str(RAW_DATA_DIR / "exoplanets_raw.csv"),
        "processed_file": str(PROCESSED_DATA_DIR / "exoplanets_processed.csv"),
        "sample_file": str(DATA_DIR / "exoplanets_sample.csv")
    }
}

# Model configuration
MODEL_CONFIG = {
    "random_forest": {
        "path": str(MODELS_DIR / "random_forest_model.pkl"),
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    },
    "xgboost": {
        "path": str(MODELS_DIR / "xgboost_model.pkl"),
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": 42
    },
    "scaler_path": str(MODELS_DIR / "scaler.pkl"),
    "encoder_path": str(MODELS_DIR / "encoder.pkl")
}

# Flask configuration
FLASK_CONFIG = {
    "host": os.getenv("HOST", "0.0.0.0"),
    "port": int(os.getenv("PORT", 5000)),
    "debug": not IS_PRODUCTION,
    "secret_key": os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
}

# Data collection sources
DATA_SOURCES = {
    "nasa_archive": "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=PS",
    "kaggle": "https://www.kaggle.com/datasets/kevinengel/exoplanet-database"
}

# Feature columns
PLANET_FEATURES = [
    "radius", "mass", "density", "surface_temp", 
    "orbital_period", "distance_from_star"
]

STAR_FEATURES = [
    "star_type", "star_luminosity", "star_temp", "metallicity"
]

ALL_FEATURES = PLANET_FEATURES + STAR_FEATURES

# Target variable
TARGET_VARIABLE = "habitability_class"  # Options: "habitability_class" or "habitability_score"

# Train/test split
TRAIN_TEST_SPLIT = 0.8
RANDOM_STATE = 42

# Habitability thresholds
HABITABILITY_THRESHOLDS = {
    "high": 0.75,
    "medium": 0.50,
    "low": 0.25
}

