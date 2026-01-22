"""
Database utility functions for ExoHabitatAI
Supports both PostgreSQL and CSV storage
Compatible with Heroku/Render DATABASE_URL
"""
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import DATABASE_CONFIG

class DatabaseManager:
    """
    Manages database operations for ExoHabitatAI
    Supports PostgreSQL (cloud) and CSV (local) storage
    """
    
    def __init__(self):
        self.db_config = DATABASE_CONFIG
        self.db_type = DATABASE_CONFIG["type"]
        self._engine = None
    
    def _get_connection_string(self):
        """
        Get PostgreSQL connection string
        Supports DATABASE_URL format from Heroku/Render
        """
        # Use full URL if available
        if self.db_config.get("url"):
            return self.db_config["url"]
        
        # Build from individual settings
        db = self.db_config["postgresql"]
        return (
            f"postgresql://{db['user']}:{db['password']}"
            f"@{db['host']}:{db['port']}/{db['database']}"
        )
    
    def _get_engine(self):
        """
        Get SQLAlchemy engine (lazy initialization)
        """
        if self._engine is None:
            try:
                from sqlalchemy import create_engine
                connection_string = self._get_connection_string()
                self._engine = create_engine(connection_string, pool_pre_ping=True)
            except Exception as e:
                print(f"Error creating database engine: {e}")
                return None
        return self._engine
    
    def test_connection(self):
        """
        Test database connection
        """
        if self.db_type != "postgresql":
            return {"status": "ok", "type": "csv"}
        
        try:
            engine = self._get_engine()
            if engine:
                from sqlalchemy import text
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                return {"status": "ok", "type": "postgresql"}
        except Exception as e:
            return {"status": "error", "type": "postgresql", "error": str(e)}
    
    def load_data(self, source="processed"):
        """
        Load data from database or CSV
        """
        if self.db_type == "postgresql":
            return self._load_from_postgresql(source)
        else:
            return self._load_from_csv(source)
    
    def _load_from_csv(self, source):
        """
        Load data from CSV file
        """
        if source == "raw":
            file_path = self.db_config["csv"]["raw_file"]
        elif source == "processed":
            file_path = self.db_config["csv"]["processed_file"]
        else:
            raise ValueError("source must be 'raw' or 'processed'")
        
        try:
            df = pd.read_csv(file_path, low_memory=False)
            print(f"Loaded {len(df)} records from {file_path}")
            return df
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            print("Generating sample data...")
            return self._generate_sample_data()
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return self._generate_sample_data()
    
    def _load_from_postgresql(self, table_name):
        """
        Load data from PostgreSQL database
        """
        try:
            engine = self._get_engine()
            if engine is None:
                print("Database engine not available, generating sample data")
                return self._generate_sample_data()
            
            # Check if table exists
            from sqlalchemy import inspect
            inspector = inspect(engine)
            if table_name not in inspector.get_table_names() and 'exoplanets' not in inspector.get_table_names():
                print(f"Table {table_name} not found, initializing with sample data...")
                sample_df = self._generate_sample_data()
                self._save_to_postgresql(sample_df, "exoplanets")
                return sample_df
            
            # Try to load from table
            actual_table = "exoplanets" if "exoplanets" in inspector.get_table_names() else table_name
            query = f"SELECT * FROM {actual_table}"
            df = pd.read_sql(query, engine)
            
            if len(df) == 0:
                print("Table empty, generating sample data...")
                sample_df = self._generate_sample_data()
                self._save_to_postgresql(sample_df, "exoplanets")
                return sample_df
            
            print(f"Loaded {len(df)} records from PostgreSQL table: {actual_table}")
            return df
            
        except Exception as e:
            print(f"Error loading from PostgreSQL: {e}")
            print("Generating sample data...")
            return self._generate_sample_data()
    
    def _generate_sample_data(self):
        """
        Generate sample exoplanet data when no data source is available
        """
        import numpy as np
        
        np.random.seed(42)
        n_samples = 500
        
        # Star types with realistic distribution
        star_types = np.random.choice(['G', 'K', 'M', 'F', 'A'], n_samples, p=[0.3, 0.3, 0.25, 0.1, 0.05])
        
        # Generate realistic data
        data = {
            'pl_name': [f'Kepler-{i}b' for i in range(1, n_samples + 1)],
            'hostname': [f'Kepler-{i}' for i in range(1, n_samples + 1)],
            'radius': np.random.uniform(0.5, 15, n_samples),
            'mass': np.random.uniform(0.1, 100, n_samples),
            'density': np.random.uniform(0.5, 10, n_samples),
            'surface_temp': np.random.uniform(200, 800, n_samples),
            'orbital_period': np.random.uniform(1, 500, n_samples),
            'distance_from_star': np.random.uniform(0.01, 5, n_samples),
            'star_type': star_types,
            'star_temp': np.random.uniform(3000, 8000, n_samples),
            'star_luminosity': np.random.uniform(0.01, 50, n_samples),
            'metallicity': np.random.uniform(-0.5, 0.5, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Calculate habitability scores
        df['habitability_score'] = df.apply(self._calculate_habitability, axis=1)
        df['habitability_class'] = df['habitability_score'].apply(
            lambda x: 'High' if x > 0.7 else ('Medium' if x > 0.4 else 'Low')
        )
        
        # Sort by habitability score
        df = df.sort_values('habitability_score', ascending=False).reset_index(drop=True)
        df['rank'] = range(1, len(df) + 1)
        
        print(f"Generated {len(df)} sample exoplanets")
        return df
    
    def _calculate_habitability(self, row):
        """Calculate habitability score for a single planet"""
        import numpy as np
        
        # Temperature factor (optimal 250-350K)
        temp = row.get('surface_temp', 300)
        temp_factor = 1.0 - min(abs(temp - 300) / 200, 1.0)
        
        # Size factor (Earth-like: 0.8-2.0 radii optimal)
        radius = row.get('radius', 1)
        size_factor = 1.0 - min(abs(radius - 1.2) / 2.0, 1.0)
        
        # Distance factor (habitable zone ~0.5-2 AU for Sun-like)
        distance = row.get('distance_from_star', 1)
        distance_factor = 1.0 - min(abs(distance - 1.0) / 1.5, 1.0)
        
        # Star type factor
        star_type = row.get('star_type', 'G')
        star_factors = {'G': 1.0, 'K': 0.9, 'F': 0.8, 'M': 0.6, 'A': 0.4}
        star_factor = star_factors.get(star_type, 0.5)
        
        # Combined score
        score = (0.35 * temp_factor + 0.25 * size_factor + 
                 0.25 * distance_factor + 0.15 * star_factor)
        
        return max(0.0, min(1.0, score))

    def save_data(self, df, destination="processed", table_name=None):
        """
        Save data to database or CSV
        """
        if self.db_type == "postgresql":
            return self._save_to_postgresql(df, table_name or destination)
        else:
            return self._save_to_csv(df, destination)
    
    def _save_to_csv(self, df, destination):
        """
        Save data to CSV file
        """
        if destination == "raw":
            file_path = self.db_config["csv"]["raw_file"]
        elif destination == "processed":
            file_path = self.db_config["csv"]["processed_file"]
        else:
            file_path = destination
        
        try:
            # Ensure directory exists
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(file_path, index=False)
            print(f"Saved {len(df)} records to {file_path}")
            return True
        except Exception as e:
            print(f"Error saving CSV: {e}")
            return False
    
    def _save_to_postgresql(self, df, table_name):
        """
        Save data to PostgreSQL database
        """
        try:
            engine = self._get_engine()
            if engine is None:
                print("Database engine not available")
                return False
            
            df.to_sql(table_name, engine, if_exists='replace', index=False)
            
            print(f"Saved {len(df)} records to PostgreSQL table: {table_name}")
            return True
            
        except Exception as e:
            print(f"Error saving to PostgreSQL: {e}")
            return False
    
    def init_database(self, df=None):
        """
        Initialize database with data (for cloud deployment)
        """
        if self.db_type != "postgresql":
            print("Database initialization only needed for PostgreSQL")
            return True
        
        try:
            # Load from CSV if no data provided
            if df is None:
                df = self._load_from_csv("processed")
                if df is None:
                    print("No data available to initialize database")
                    return False
            
            # Save to PostgreSQL
            return self._save_to_postgresql(df, "exoplanets")
            
        except Exception as e:
            print(f"Error initializing database: {e}")
            return False

