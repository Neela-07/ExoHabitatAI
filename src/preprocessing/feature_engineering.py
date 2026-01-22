"""
Module 2: Feature Engineering
Encodes categorical features using one-hot encoding
Creates engineered features:
- Habitability Score Index (HSI)
- Stellar Compatibility Index (SCI)
Normalizes numerical features
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DATA_DIR, MODELS_DIR
from src.utils.database import DatabaseManager
import pickle

class FeatureEngineer:
    """
    Performs feature engineering on exoplanet data
    """
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.star_type_mapping = {
            'O': 0, 'B': 1, 'A': 2, 'F': 3, 
            'G': 4, 'K': 5, 'M': 6
        }
    
    def engineer_features(self, df):
        """
        Main feature engineering pipeline
        """
        print("Starting feature engineering process...")
        
        # Step 0: Calculate derived physical properties
        df = self.calculate_derived_properties(df)
        
        # Step 1: Create Habitability Score Index (HSI)
        df = self.create_habitability_score_index(df)
        
        # Step 2: Create Stellar Compatibility Index (SCI)
        df = self.create_stellar_compatibility_index(df)
        
        # Step 3: Create additional derived features
        df = self.create_derived_features(df)
        
        # Step 4: Encode categorical features
        df = self.encode_categorical_features(df)
        
        # Step 5: Create habitability class
        df = self.create_habitability_class(df)
        
        print("Feature engineering completed!")
        return df
    
    def calculate_derived_properties(self, df):
        """
        Calculate derived physical properties from available data
        - Density from mass and radius
        - Stellar luminosity from radius and temperature
        """
        print("\nCalculating derived properties...")
        
        # Calculate planet density (g/cm³)
        # density = mass / ((4/3) * π * radius³)
        # mass in Earth masses, radius in Earth radii -> density in g/cm³
        if 'mass' in df.columns and 'radius' in df.columns:
            # Earth density = 5.51 g/cm³
            # density = (mass_earth / radius_earth³) * 5.51
            df['density'] = (df['mass'] / (df['radius'] ** 3)) * 5.51
            # Cap extreme values
            df['density'] = df['density'].clip(0.1, 50.0)
            print(f"  Calculated density from mass and radius")
            print(f"    Range: {df['density'].min():.2f} to {df['density'].max():.2f} g/cm³")
        else:
            df['density'] = 5.51  # Earth density as default
        
        # Calculate stellar luminosity (Solar units)
        # L = R² * (T/T_sun)⁴
        if 'star_radius' in df.columns and 'star_temp' in df.columns:
            T_sun = 5772  # Kelvin
            df['star_luminosity'] = (df['star_radius'] ** 2) * ((df['star_temp'] / T_sun) ** 4)
            df['star_luminosity'] = df['star_luminosity'].clip(0.0001, 1000.0)
            print(f"  Calculated stellar luminosity from radius and temperature")
            print(f"    Range: {df['star_luminosity'].min():.4f} to {df['star_luminosity'].max():.2f} L_sun")
        else:
            df['star_luminosity'] = 1.0  # Solar luminosity as default
        
        return df
    
    def create_habitability_score_index(self, df):
        """
        Create Habitability Score Index (HSI) based on key planetary parameters
        HSI considers:
        - Temperature zone (habitable zone)
        - Surface temperature (comfortable range: 273-323 K)
        - Planetary radius (Earth-like: 0.8-1.5 R_Earth)
        - Density (rocky planets: 4-6 g/cm³)
        """
        print("\nCreating Habitability Score Index (HSI)...")
        
        # Normalize each factor to 0-1 scale
        # Temperature factor (optimal around 288K / Earth temperature)
        if 'surface_temp' in df.columns:
            optimal_temp = 288  # Kelvin (Earth average)
            temp_factor = 1 - np.abs(df['surface_temp'] - optimal_temp) / 200  # Range ±200K
            temp_factor = np.clip(temp_factor, 0, 1)
        else:
            temp_factor = 0.5
        
        # Radius factor (optimal around 1 Earth radius)
        if 'radius' in df.columns:
            optimal_radius = 1.0
            radius_factor = 1 - np.abs(df['radius'] - optimal_radius) / 2  # Range ±2 R_Earth
            radius_factor = np.clip(radius_factor, 0, 1)
        else:
            radius_factor = 0.5
        
        # Density factor (optimal for rocky planets: 4-6 g/cm³)
        if 'density' in df.columns:
            optimal_density_min, optimal_density_max = 4, 6
            density_factor = np.where(
                (df['density'] >= optimal_density_min) & (df['density'] <= optimal_density_max),
                1.0,
                1 - np.abs(df['density'] - 5) / 5  # Penalty for deviation
            )
            density_factor = np.clip(density_factor, 0, 1)
        else:
            density_factor = 0.5
        
        # Distance factor (habitable zone - simplified)
        if 'distance_from_star' in df.columns and 'star_luminosity' in df.columns:
            # Simplified habitable zone calculation
            habitable_zone_inner = np.sqrt(df['star_luminosity'] / 1.1)
            habitable_zone_outer = np.sqrt(df['star_luminosity'] / 0.53)
            
            distance_factor = np.where(
                (df['distance_from_star'] >= habitable_zone_inner) & 
                (df['distance_from_star'] <= habitable_zone_outer),
                1.0,
                0.5  # Penalty for being outside HZ
            )
        else:
            distance_factor = 0.5
        
        # Calculate HSI as weighted average
        df['habitability_score_index'] = (
            0.3 * temp_factor +
            0.25 * radius_factor +
            0.25 * density_factor +
            0.2 * distance_factor
        )
        
        print(f"  HSI range: {df['habitability_score_index'].min():.3f} - {df['habitability_score_index'].max():.3f}")
        
        return df
    
    def create_stellar_compatibility_index(self, df):
        """
        Create Stellar Compatibility Index (SCI) to measure host star influence
        SCI considers:
        - Star type (G-type stars are most compatible)
        - Star temperature (optimal: 5000-6000 K)
        - Metallicity (solar-like: ~0)
        - Stability (longer orbital periods preferred)
        """
        print("\nCreating Stellar Compatibility Index (SCI)...")
        
        # Star type factor (G-type is optimal)
        if 'star_type' in df.columns:
            star_type_numeric = df['star_type'].map(self.star_type_mapping).fillna(4)
            # G-type = 4, optimal
            star_type_factor = 1 - np.abs(star_type_numeric - 4) / 6
            star_type_factor = np.clip(star_type_factor, 0, 1)
        else:
            star_type_factor = 0.5
        
        # Star temperature factor (optimal: 5500K)
        if 'star_temp' in df.columns:
            optimal_star_temp = 5778  # Sun's temperature
            star_temp_factor = 1 - np.abs(df['star_temp'] - optimal_star_temp) / 3000
            star_temp_factor = np.clip(star_temp_factor, 0, 1)
        else:
            star_temp_factor = 0.5
        
        # Metallicity factor (solar-like: 0)
        if 'metallicity' in df.columns:
            metallicity_factor = 1 - np.abs(df['metallicity']) / 1.0
            metallicity_factor = np.clip(metallicity_factor, 0, 1)
        else:
            metallicity_factor = 0.5
        
        # Orbital period factor (longer periods = more stable)
        if 'orbital_period' in df.columns:
            # Optimal around 365 days (Earth-like)
            optimal_period = 365
            period_factor = 1 - np.abs(np.log10(df['orbital_period'] + 1) - np.log10(optimal_period + 1)) / 2
            period_factor = np.clip(period_factor, 0, 1)
        else:
            period_factor = 0.5
        
        # Calculate SCI as weighted average
        df['stellar_compatibility_index'] = (
            0.3 * star_type_factor +
            0.3 * star_temp_factor +
            0.2 * metallicity_factor +
            0.2 * period_factor
        )
        
        print(f"  SCI range: {df['stellar_compatibility_index'].min():.3f} - {df['stellar_compatibility_index'].max():.3f}")
        
        return df
    
    def create_derived_features(self, df):
        """
        Create additional derived features
        """
        print("\nCreating derived features...")
        
        # Planet-to-star ratio
        if 'radius' in df.columns and 'distance_from_star' in df.columns:
            df['radius_distance_ratio'] = df['radius'] / (df['distance_from_star'] + 1e-6)
        
        # Mass-radius ratio
        if 'mass' in df.columns and 'radius' in df.columns:
            df['mass_radius_ratio'] = df['mass'] / (df['radius'] + 1e-6)
        
        # Temperature-pressure proxy (simplified)
        if 'surface_temp' in df.columns and 'density' in df.columns:
            df['temp_density_interaction'] = df['surface_temp'] * df['density']
        
        # Combined habitability score
        if 'habitability_score_index' in df.columns and 'stellar_compatibility_index' in df.columns:
            df['combined_habitability_score'] = (
                0.6 * df['habitability_score_index'] +
                0.4 * df['stellar_compatibility_index']
            )
        
        print(f"  Created {4} derived features")
        
        return df
    
    def encode_categorical_features(self, df):
        """
        Encode categorical features using one-hot encoding
        """
        print("\nEncoding categorical features...")
        
        # One-hot encode star_type
        if 'star_type' in df.columns:
            star_type_dummies = pd.get_dummies(df['star_type'], prefix='star_type')
            df = pd.concat([df, star_type_dummies], axis=1)
            print(f"  Created {len(star_type_dummies.columns)} one-hot encoded features for star_type")
        
        return df
    
    def create_habitability_class(self, df):
        """
        Create habitability class labels based on scores
        """
        print("\nCreating habitability class labels...")
        
        if 'combined_habitability_score' in df.columns:
            df['habitability_class'] = pd.cut(
                df['combined_habitability_score'],
                bins=[0, 0.25, 0.50, 0.75, 1.0],
                labels=['Non-Habitable', 'Low', 'Medium', 'High'],
                include_lowest=True
            )
        elif 'habitability_score_index' in df.columns:
            df['habitability_class'] = pd.cut(
                df['habitability_score_index'],
                bins=[0, 0.25, 0.50, 0.75, 1.0],
                labels=['Non-Habitable', 'Low', 'Medium', 'High'],
                include_lowest=True
            )
        
        if 'habitability_class' in df.columns:
            print("\nHabitability class distribution:")
            print(df['habitability_class'].value_counts())
        
        return df
    
    def normalize_features(self, df, feature_columns, fit=True):
        """
        Normalize numerical features using StandardScaler
        """
        print(f"\nNormalizing {len(feature_columns)} features...")
        
        if fit:
            df[feature_columns] = self.scaler.fit_transform(df[feature_columns])
            # Save scaler
            scaler_path = MODELS_DIR / "scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"  Saved scaler to {scaler_path}")
        else:
            # Load existing scaler
            scaler_path = MODELS_DIR / "scaler.pkl"
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                df[feature_columns] = self.scaler.transform(df[feature_columns])
        
        return df

def main():
    """
    Main function for feature engineering
    """
    engineer = FeatureEngineer()
    
    # Load cleaned data
    df = engineer.db_manager.load_data(source="processed")
    
    if df is None:
        print("Error: Could not load processed data. Please run data cleaning first.")
        return
    
    # Engineer features
    engineered_df = engineer.engineer_features(df)
    
    # Save engineered data
    engineer.db_manager.save_data(engineered_df, destination="processed")
    
    print("\nFeature engineering completed successfully!")

if __name__ == "__main__":
    main()

