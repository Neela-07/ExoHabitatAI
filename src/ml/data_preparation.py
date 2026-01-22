"""
Module 3: Machine Learning Dataset Preparation
Split data into training and testing sets (80:20)
Select features for model input based on correlation with habitability
Define target variable: Habitability class or habitability score
Create data pipelines with scaling, encoding, and feature selection
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sys
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import TRAIN_TEST_SPLIT, RANDOM_STATE, MODELS_DIR, TARGET_VARIABLE
from src.utils.database import DatabaseManager

class MLDataPreparation:
    """
    Prepares data for machine learning models
    """
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.target_column = TARGET_VARIABLE
        
    def prepare_data(self, test_size=0.2, random_state=42):
        """
        Main data preparation pipeline
        """
        print("Starting ML data preparation...")
        
        # Load processed data
        df = self.db_manager.load_data(source="processed")
        
        if df is None:
            print("Error: Could not load processed data.")
            return None, None, None, None
        
        # Step 1: Select features
        feature_columns = self.select_features(df)
        
        # Step 2: Handle target variable
        df = self.prepare_target_variable(df)
        
        # Step 3: Prepare feature matrix and target vector
        X = df[feature_columns].copy()
        y = df[self.target_column].copy()
        
        # Step 4: Handle missing values in features
        X = X.fillna(X.median())
        
        # Step 5: Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if y.dtype == 'object' else None
        )
        
        print(f"\nData split:")
        print(f"  Training set: {len(X_train)} samples")
        print(f"  Testing set: {len(X_test)} samples")
        print(f"  Features: {len(feature_columns)}")
        
        # Step 6: Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Save scaler
        scaler_path = MODELS_DIR / "scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"  Saved scaler to {scaler_path}")
        
        # Step 7: Encode target if categorical
        if y_train.dtype == 'object':
            y_train_encoded = self.label_encoder.fit_transform(y_train)
            y_test_encoded = self.label_encoder.transform(y_test)
            
            # Save encoder
            encoder_path = MODELS_DIR / "label_encoder.pkl"
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.label_encoder, f)
            print(f"  Saved label encoder to {encoder_path}")
            
            return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def select_features(self, df):
        """
        Select features based on correlation with habitability
        """
        print("\nSelecting features based on correlation...")
        
        # Base features (always include)
        base_features = [
            'radius', 'mass', 'density', 'surface_temp',
            'orbital_period', 'distance_from_star',
            'star_luminosity', 'star_temp', 'metallicity'
        ]
        
        # Engineered features (include if available)
        engineered_features = [
            'habitability_score_index', 'stellar_compatibility_index',
            'combined_habitability_score', 'radius_distance_ratio',
            'mass_radius_ratio', 'temp_density_interaction'
        ]
        
        # Star type one-hot encoded features
        star_type_features = [col for col in df.columns if col.startswith('star_type_')]
        
        # Combine all features
        all_features = base_features + engineered_features + star_type_features
        
        # Filter to features that exist in dataframe
        available_features = [f for f in all_features if f in df.columns]
        
        # Calculate correlation with target if numeric
        if 'combined_habitability_score' in df.columns:
            correlations = df[available_features].corrwith(df['combined_habitability_score']).abs().sort_values(ascending=False)
            print("\nTop features by correlation with habitability:")
            for feat, corr in correlations.head(10).items():
                print(f"  {feat}: {corr:.3f}")
        
        # Remove target variable if present
        target_columns = ['habitability_class', 'combined_habitability_score', 
                         'habitability_score_index', 'stellar_compatibility_index']
        available_features = [f for f in available_features if f not in target_columns]
        
        self.feature_columns = available_features
        print(f"\nSelected {len(available_features)} features for modeling")
        
        return available_features
    
    def prepare_target_variable(self, df):
        """
        Prepare target variable (habitability class or score)
        """
        print(f"\nPreparing target variable: {self.target_column}")
        
        if self.target_column == "habitability_class":
            # Ensure habitability_class exists
            if 'habitability_class' not in df.columns:
                if 'combined_habitability_score' in df.columns:
                    df['habitability_class'] = pd.cut(
                        df['combined_habitability_score'],
                        bins=[0, 0.25, 0.50, 0.75, 1.0],
                        labels=['Non-Habitable', 'Low', 'Medium', 'High'],
                        include_lowest=True
                    )
                else:
                    print("Warning: Cannot create habitability_class. Using score instead.")
                    self.target_column = "combined_habitability_score"
        
        elif self.target_column == "combined_habitability_score":
            # Ensure combined_habitability_score exists
            if 'combined_habitability_score' not in df.columns:
                if 'habitability_score_index' in df.columns:
                    if 'stellar_compatibility_index' in df.columns:
                        df['combined_habitability_score'] = (
                            0.6 * df['habitability_score_index'] +
                            0.4 * df['stellar_compatibility_index']
                        )
                    else:
                        df['combined_habitability_score'] = df['habitability_score_index']
        
        if self.target_column not in df.columns:
            print(f"Error: Target variable '{self.target_column}' not found in data.")
            return None
        
        print(f"  Target variable '{self.target_column}' prepared")
        print(f"  Distribution:")
        print(df[self.target_column].value_counts())
        
        return df
    
    def get_feature_names(self):
        """
        Get selected feature names
        """
        return self.feature_columns

def main():
    """
    Main function for data preparation
    """
    prep = MLDataPreparation()
    
    # Prepare data
    X_train, X_test, y_train, y_test = prep.prepare_data(
        test_size=(1 - TRAIN_TEST_SPLIT),
        random_state=RANDOM_STATE
    )
    
    if X_train is not None:
        print("\nML data preparation completed successfully!")
        print(f"  Training shape: {X_train.shape}")
        print(f"  Testing shape: {X_test.shape}")
    else:
        print("\nML data preparation failed.")

if __name__ == "__main__":
    main()

