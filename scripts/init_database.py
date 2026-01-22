"""
Initialize PostgreSQL database with exoplanet data
Run this after deploying to Render to populate the database
Usage: python scripts/init_database.py
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from config import DATABASE_CONFIG, DATA_DIR
from src.utils.database import DatabaseManager

def init_database():
    """
    Initialize database with sample exoplanet data
    """
    print("="*60)
    print("ExoHabitatAI - Database Initialization")
    print("="*60)
    
    db = DatabaseManager()
    
    # Check database type
    print(f"Database type: {db.db_type}")
    
    if db.db_type != "postgresql":
        print("No PostgreSQL database configured.")
        print("Set DATABASE_URL environment variable for cloud deployment.")
        return False
    
    # Test connection
    status = db.test_connection()
    print(f"Connection status: {status}")
    
    if status.get("status") != "ok":
        print("Failed to connect to database!")
        return False
    
    # Try to load local CSV data
    csv_path = DATA_DIR / "processed" / "exoplanets_processed.csv"
    
    if csv_path.exists():
        print(f"Loading data from: {csv_path}")
        df = pd.read_csv(csv_path, low_memory=False)
        print(f"Loaded {len(df)} records")
    else:
        print("No local CSV file found. Creating sample data...")
        df = create_sample_data()
    
    # Initialize database
    print("Saving to PostgreSQL...")
    success = db.init_database(df)
    
    if success:
        print("✅ Database initialized successfully!")
    else:
        print("❌ Failed to initialize database")
    
    return success

def create_sample_data():
    """
    Create sample exoplanet data for testing
    """
    import numpy as np
    
    np.random.seed(42)
    n_samples = 1000
    
    # Generate sample data
    data = {
        'pl_name': [f'Kepler-{i}b' for i in range(1, n_samples + 1)],
        'hostname': [f'Kepler-{i}' for i in range(1, n_samples + 1)],
        'pl_rade': np.random.uniform(0.5, 15, n_samples),  # Earth radii
        'pl_masse': np.random.uniform(0.1, 100, n_samples),  # Earth masses
        'pl_dens': np.random.uniform(0.5, 10, n_samples),  # g/cm³
        'pl_eqt': np.random.uniform(200, 1000, n_samples),  # K
        'pl_orbper': np.random.uniform(1, 1000, n_samples),  # days
        'pl_orbsmax': np.random.uniform(0.01, 10, n_samples),  # AU
        'st_spectype': np.random.choice(['G', 'K', 'M', 'F', 'A'], n_samples),
        'st_teff': np.random.uniform(3000, 10000, n_samples),  # K
        'st_lum': np.random.uniform(0.01, 100, n_samples),  # Solar
        'st_met': np.random.uniform(-1, 1, n_samples),  # dex
        'habitability_score': np.random.uniform(0, 1, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Sort by habitability score
    df = df.sort_values('habitability_score', ascending=False).reset_index(drop=True)
    
    print(f"Created {len(df)} sample exoplanets")
    return df

if __name__ == "__main__":
    init_database()
