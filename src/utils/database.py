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
            return None
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return None
    
    def _load_from_postgresql(self, table_name):
        """
        Load data from PostgreSQL database
        """
        try:
            engine = self._get_engine()
            if engine is None:
                print("Database engine not available, falling back to CSV")
                return self._load_from_csv("processed")
            
            # Try to load from table
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql(query, engine)
            
            print(f"Loaded {len(df)} records from PostgreSQL table: {table_name}")
            return df
            
        except Exception as e:
            print(f"Error loading from PostgreSQL: {e}")
            print("Falling back to CSV data source")
            return self._load_from_csv("processed")
    
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

