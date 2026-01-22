"""
Module 1: Data Collection and Management
Collects exoplanet datasets from NASA Exoplanet Archive and Kaggle
Stores data in PostgreSQL or CSV format
Validates schema and ensures completeness
"""
import pandas as pd
import requests
import os
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import DATABASE_CONFIG, RAW_DATA_DIR

class ExoplanetDataCollector:
    """
    Collects and manages exoplanet data from various sources
    """
    
    def __init__(self):
        self.raw_data_dir = RAW_DATA_DIR
        self.db_config = DATABASE_CONFIG
        
    def collect_from_csv(self, csv_path):
        """
        Load existing CSV dataset
        """
        try:
            df = pd.read_csv(csv_path)
            print(f"Loaded dataset with {len(df)} records")
            return df
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return None
    
    def collect_from_nasa_api(self, limit=1000):
        """
        Collect data from NASA Exoplanet Archive API
        Note: This is a placeholder for API integration
        """
        try:
            # NASA Exoplanet Archive API endpoint
            url = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"
            params = {
                "table": "exoplanets",
                "format": "csv",
                "select": "pl_name,pl_radiusj,pl_massj,pl_dens,pl_orbper,pl_orbsmax,"
                         "st_spectype,st_lum,st_teff,st_met"
            }
            
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                # Save raw data
                raw_file = self.raw_data_dir / "nasa_exoplanets_raw.csv"
                with open(raw_file, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                
                df = pd.read_csv(raw_file)
                print(f"Collected {len(df)} records from NASA Archive")
                return df
            else:
                print(f"API request failed with status {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error collecting from NASA API: {e}")
            return None
    
    def create_sample_dataset(self):
        """
        Create a sample dataset if external sources are unavailable
        For demonstration purposes
        """
        import numpy as np
        
        np.random.seed(42)
        n_samples = 5000
        
        data = {
            # Planetary parameters
            "planet_name": [f"Planet_{i}" for i in range(n_samples)],
            "radius": np.random.uniform(0.5, 25, n_samples),
            "mass": np.random.uniform(0.1, 20, n_samples),
            "density": np.random.uniform(1, 15, n_samples),
            "surface_temp": np.random.uniform(150, 600, n_samples),
            "orbital_period": np.random.uniform(0.5, 10000, n_samples),
            "distance_from_star": np.random.uniform(0.01, 100, n_samples),
            
            # Stellar parameters
            "star_type": np.random.choice(['O', 'B', 'A', 'F', 'G', 'K', 'M'], n_samples),
            "star_luminosity": np.random.uniform(0.001, 1000, n_samples),
            "star_temp": np.random.uniform(2000, 50000, n_samples),
            "metallicity": np.random.uniform(-1.0, 1.0, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Save to CSV
        output_file = self.raw_data_dir / "exoplanets_raw.csv"
        df.to_csv(output_file, index=False)
        print(f"Created sample dataset with {len(df)} records")
        
        return df
    
    def validate_schema(self, df):
        """
        Validate dataset schema and completeness
        """
        required_columns = [
            "radius", "mass", "density", "surface_temp",
            "orbital_period", "distance_from_star",
            "star_type", "star_luminosity", "star_temp", "metallicity"
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Warning: Missing columns: {missing_columns}")
            return False
        
        # Check for empty dataset
        if df.empty:
            print("Error: Dataset is empty")
            return False
        
        # Check completeness
        completeness = (1 - df[required_columns].isnull().sum() / len(df)) * 100
        print("\nData Completeness:")
        for col in required_columns:
            print(f"  {col}: {completeness[col]:.2f}%")
        
        return True
    
    def store_in_csv(self, df, filename="exoplanets_raw.csv"):
        """
        Store data in CSV format
        """
        output_file = self.raw_data_dir / filename
        df.to_csv(output_file, index=False)
        print(f"Data stored in CSV: {output_file}")
        return output_file
    
    def store_in_postgresql(self, df, table_name="exoplanets_raw"):
        """
        Store data in PostgreSQL database
        """
        if self.db_config["type"] != "postgresql":
            print("PostgreSQL not configured. Using CSV instead.")
            return self.store_in_csv(df)
        
        try:
            from sqlalchemy import create_engine
            from config import DATABASE_CONFIG
            
            db_config = DATABASE_CONFIG["postgresql"]
            connection_string = (
                f"postgresql://{db_config['user']}:{db_config['password']}"
                f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            )
            
            engine = create_engine(connection_string)
            df.to_sql(table_name, engine, if_exists='replace', index=False)
            print(f"Data stored in PostgreSQL table: {table_name}")
            return True
            
        except Exception as e:
            print(f"Error storing in PostgreSQL: {e}")
            return False

def main():
    """
    Main function for data collection
    """
    collector = ExoplanetDataCollector()
    
    # Try to load existing dataset first
    existing_csv = Path(__file__).parent.parent.parent / "data" / "merged_exoplanet_dataset (1).csv"
    
    if existing_csv.exists():
        print(f"Loading existing dataset from {existing_csv}")
        df = collector.collect_from_csv(existing_csv)
    else:
        print("No existing dataset found. Creating sample dataset...")
        df = collector.create_sample_dataset()
    
    if df is not None:
        # Validate schema
        if collector.validate_schema(df):
            # Store data
            collector.store_in_csv(df)
            print("\nData collection completed successfully!")
        else:
            print("\nData validation failed. Creating sample dataset instead.")
            df = collector.create_sample_dataset()
            if df is not None:
                collector.store_in_csv(df)
                print("\nSample data collection completed successfully!")
    else:
        print("Data collection failed. Creating sample dataset...")
        df = collector.create_sample_dataset()
        if df is not None:
            collector.store_in_csv(df)
            print("\nSample data collection completed successfully!")

if __name__ == "__main__":
    main()

