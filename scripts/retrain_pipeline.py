"""
Retrain Pipeline - Processes new data and retrains models
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings('ignore')

from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR,
    MODEL_CONFIG, RANDOM_STATE
)

def load_and_clean_data():
    """Load and clean the raw data"""
    print("="*60)
    print("STEP 1: Loading and Cleaning Data")
    print("="*60)
    
    raw_file = RAW_DATA_DIR / "exoplanets_raw.csv"
    df = pd.read_csv(raw_file, low_memory=False)
    print(f"Loaded {len(df)} records from {raw_file}")
    
    # Column mapping for NASA data
    column_mapping = {
        'pl_rade': 'radius',
        'pl_bmasse': 'mass',
        'pl_orbper': 'orbital_period',
        'pl_orbsmax': 'distance_from_star',
        'pl_eqt': 'surface_temp',
        'st_teff': 'star_temp',
        'st_rad': 'star_radius',
        'st_mass': 'star_mass',
        'st_met': 'metallicity',
        'st_logg': 'star_logg',
        'pl_name': 'planet_name',
        'hostname': 'host_star'
    }
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    
    # Remove duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"Removing {duplicates} duplicate rows...")
        df = df.drop_duplicates().reset_index(drop=True)
    
    # Numerical columns to clean
    numerical_cols = ['radius', 'mass', 'surface_temp', 
                     'orbital_period', 'distance_from_star', 
                     'star_radius', 'star_temp', 'star_mass', 'metallicity']
    
    for col in numerical_cols:
        if col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                median_value = df[col].median()
                df[col].fillna(median_value, inplace=True)
                print(f"  {col}: Filled {missing_count} missing values with median {median_value:.2f}")
    
    # Extract star type
    if 'st_spectype' in df.columns:
        def extract_star_type(spectype):
            if pd.isna(spectype):
                return 'G'
            spec_str = str(spectype).strip().upper()
            if len(spec_str) > 0 and spec_str[0] in ['O', 'B', 'A', 'F', 'G', 'K', 'M']:
                return spec_str[0]
            return 'G'
        df['star_type'] = df['st_spectype'].apply(extract_star_type)
    else:
        df['star_type'] = 'G'
    
    print(f"Cleaned data: {len(df)} records")
    return df


def engineer_features(df):
    """Engineer features for ML"""
    print("\n" + "="*60)
    print("STEP 2: Feature Engineering")
    print("="*60)
    
    star_type_mapping = {'O': 0, 'B': 1, 'A': 2, 'F': 3, 'G': 4, 'K': 5, 'M': 6}
    
    # Calculate density
    if 'mass' in df.columns and 'radius' in df.columns:
        df['density'] = (df['mass'] / (df['radius'] ** 3 + 1e-6)) * 5.51
        df['density'] = df['density'].clip(0.1, 50.0)
        print(f"  Calculated density: range {df['density'].min():.2f} - {df['density'].max():.2f}")
    else:
        df['density'] = 5.51
    
    # Calculate stellar luminosity
    T_sun = 5772
    if 'star_radius' in df.columns and 'star_temp' in df.columns:
        df['star_luminosity'] = (df['star_radius'] ** 2) * ((df['star_temp'] / T_sun) ** 4)
        df['star_luminosity'] = df['star_luminosity'].clip(0.0001, 1000.0)
        print(f"  Calculated luminosity: range {df['star_luminosity'].min():.4f} - {df['star_luminosity'].max():.2f}")
    else:
        df['star_luminosity'] = 1.0
    
    # Habitability Score Index (HSI)
    optimal_temp = 288
    if 'surface_temp' in df.columns:
        temp_factor = 1 - np.abs(df['surface_temp'] - optimal_temp) / 200
        temp_factor = np.clip(temp_factor, 0, 1)
    else:
        temp_factor = 0.5
    
    if 'radius' in df.columns:
        radius_factor = 1 - np.abs(df['radius'] - 1.0) / 2
        radius_factor = np.clip(radius_factor, 0, 1)
    else:
        radius_factor = 0.5
    
    if 'density' in df.columns:
        density_factor = np.where(
            (df['density'] >= 4) & (df['density'] <= 6), 1.0,
            1 - np.abs(df['density'] - 5) / 5
        )
        density_factor = np.clip(density_factor, 0, 1)
    else:
        density_factor = 0.5
    
    if 'distance_from_star' in df.columns and 'star_luminosity' in df.columns:
        hz_inner = np.sqrt(df['star_luminosity'] / 1.1)
        hz_outer = np.sqrt(df['star_luminosity'] / 0.53)
        distance_factor = np.where(
            (df['distance_from_star'] >= hz_inner) & (df['distance_from_star'] <= hz_outer),
            1.0, 0.5
        )
    else:
        distance_factor = 0.5
    
    df['habitability_score_index'] = (
        0.3 * temp_factor + 0.25 * radius_factor + 
        0.25 * density_factor + 0.2 * distance_factor
    )
    print(f"  HSI range: {df['habitability_score_index'].min():.3f} - {df['habitability_score_index'].max():.3f}")
    
    # Stellar Compatibility Index (SCI)
    if 'star_type' in df.columns:
        star_type_numeric = df['star_type'].map(star_type_mapping).fillna(4)
        star_type_factor = 1 - np.abs(star_type_numeric - 4) / 6
        star_type_factor = np.clip(star_type_factor, 0, 1)
    else:
        star_type_factor = 0.5
    
    if 'star_temp' in df.columns:
        star_temp_factor = 1 - np.abs(df['star_temp'] - 5778) / 3000
        star_temp_factor = np.clip(star_temp_factor, 0, 1)
    else:
        star_temp_factor = 0.5
    
    if 'metallicity' in df.columns:
        metallicity_factor = 1 - np.abs(df['metallicity']) / 1.0
        metallicity_factor = np.clip(metallicity_factor, 0, 1)
    else:
        metallicity_factor = 0.5
    
    if 'orbital_period' in df.columns:
        period_factor = 1 - np.abs(np.log10(df['orbital_period'] + 1) - np.log10(366)) / 2
        period_factor = np.clip(period_factor, 0, 1)
    else:
        period_factor = 0.5
    
    df['stellar_compatibility_index'] = (
        0.3 * star_type_factor + 0.3 * star_temp_factor +
        0.2 * metallicity_factor + 0.2 * period_factor
    )
    print(f"  SCI range: {df['stellar_compatibility_index'].min():.3f} - {df['stellar_compatibility_index'].max():.3f}")
    
    # Combined score
    df['combined_habitability_score'] = (
        0.6 * df['habitability_score_index'] + 
        0.4 * df['stellar_compatibility_index']
    )
    print(f"  Combined score range: {df['combined_habitability_score'].min():.3f} - {df['combined_habitability_score'].max():.3f}")
    
    # Additional derived features
    if 'radius' in df.columns and 'distance_from_star' in df.columns:
        df['radius_distance_ratio'] = df['radius'] / (df['distance_from_star'] + 1e-6)
    if 'mass' in df.columns and 'radius' in df.columns:
        df['mass_radius_ratio'] = df['mass'] / (df['radius'] + 1e-6)
    if 'surface_temp' in df.columns and 'density' in df.columns:
        df['temp_density_interaction'] = df['surface_temp'] * df['density']
    
    # One-hot encode star_type
    if 'star_type' in df.columns:
        star_type_dummies = pd.get_dummies(df['star_type'], prefix='star_type')
        df = pd.concat([df, star_type_dummies], axis=1)
        print(f"  Created {len(star_type_dummies.columns)} star type features")
    
    # Create habitability class
    df['habitability_class'] = pd.cut(
        df['combined_habitability_score'],
        bins=[0, 0.25, 0.50, 0.75, 1.0],
        labels=['Non-Habitable', 'Low', 'Medium', 'High'],
        include_lowest=True
    )
    
    print("\nHabitability class distribution:")
    print(df['habitability_class'].value_counts())
    
    return df


def save_processed_data(df):
    """Save processed data"""
    processed_file = PROCESSED_DATA_DIR / "exoplanets_processed.csv"
    df.to_csv(processed_file, index=False)
    print(f"\nSaved processed data to {processed_file}")
    
    # Also save to the full data file
    full_file = Path("data/exoplanets_full.csv")
    df.to_csv(full_file, index=False)
    print(f"Saved full data to {full_file}")


def train_models(df):
    """Train ML models"""
    print("\n" + "="*60)
    print("STEP 3: Training Models")
    print("="*60)
    
    # Select features
    base_features = [
        'radius', 'mass', 'density', 'surface_temp',
        'orbital_period', 'distance_from_star',
        'star_luminosity', 'star_temp', 'metallicity'
    ]
    
    derived_features = [
        'radius_distance_ratio', 'mass_radius_ratio', 'temp_density_interaction'
    ]
    
    star_type_features = [col for col in df.columns if col.startswith('star_type_')]
    
    all_features = base_features + derived_features + star_type_features
    available_features = [f for f in all_features if f in df.columns]
    
    print(f"Using {len(available_features)} features")
    
    # Prepare data
    X = df[available_features].copy()
    y = df['habitability_class'].copy()
    
    # Remove any remaining NaN
    X = X.fillna(X.median())
    
    # Remove rows with missing target
    valid_idx = ~y.isna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    print(f"Training samples: {len(X)}")
    
    # Encode target
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
    )
    
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Testing set: {len(X_test)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    scaler_path = MODELS_DIR / "scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  Saved scaler to {scaler_path}")
    
    # Save label encoder
    encoder_path = MODELS_DIR / "label_encoder.pkl"
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"  Saved label encoder to {encoder_path}")
    
    # Save feature columns
    features_path = MODELS_DIR / "feature_columns.pkl"
    with open(features_path, 'wb') as f:
        pickle.dump(available_features, f)
    print(f"  Saved feature columns to {features_path}")
    
    # Train Random Forest
    print("\nTraining Random Forest...")
    rf_config = MODEL_CONFIG["random_forest"]
    rf_model = RandomForestClassifier(
        n_estimators=rf_config["n_estimators"],
        max_depth=rf_config["max_depth"],
        random_state=rf_config["random_state"],
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    
    rf_train_acc = rf_model.score(X_train_scaled, y_train)
    rf_test_acc = rf_model.score(X_test_scaled, y_test)
    print(f"  Random Forest - Train Accuracy: {rf_train_acc:.4f}, Test Accuracy: {rf_test_acc:.4f}")
    
    rf_path = MODELS_DIR / "random_forest_model.pkl"
    with open(rf_path, 'wb') as f:
        pickle.dump(rf_model, f)
    print(f"  Saved Random Forest to {rf_path}")
    
    # Train XGBoost
    print("\nTraining XGBoost...")
    xgb_config = MODEL_CONFIG["xgboost"]
    xgb_model = xgb.XGBClassifier(
        n_estimators=xgb_config["n_estimators"],
        max_depth=xgb_config["max_depth"],
        learning_rate=xgb_config["learning_rate"],
        random_state=xgb_config["random_state"],
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    xgb_model.fit(X_train_scaled, y_train)
    
    xgb_train_acc = xgb_model.score(X_train_scaled, y_train)
    xgb_test_acc = xgb_model.score(X_test_scaled, y_test)
    print(f"  XGBoost - Train Accuracy: {xgb_train_acc:.4f}, Test Accuracy: {xgb_test_acc:.4f}")
    
    xgb_path = MODELS_DIR / "xgboost_model.pkl"
    with open(xgb_path, 'wb') as f:
        pickle.dump(xgb_model, f)
    print(f"  Saved XGBoost to {xgb_path}")
    
    # Select best model
    if xgb_test_acc >= rf_test_acc:
        best_model = xgb_model
        best_name = "XGBoost"
        best_acc = xgb_test_acc
    else:
        best_model = rf_model
        best_name = "Random Forest"
        best_acc = rf_test_acc
    
    best_path = MODELS_DIR / "best_model.pkl"
    with open(best_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"\n  Best model: {best_name} (Accuracy: {best_acc:.4f})")
    print(f"  Saved to {best_path}")
    
    return {
        'rf_accuracy': rf_test_acc,
        'xgb_accuracy': xgb_test_acc,
        'best_model': best_name,
        'best_accuracy': best_acc
    }


def main():
    """Main execution"""
    print("="*60)
    print("ExoHabitatAI - Retrain Pipeline")
    print("="*60)
    print()
    
    # Step 1: Load and clean
    df = load_and_clean_data()
    
    # Step 2: Engineer features
    df = engineer_features(df)
    
    # Step 3: Save processed data
    save_processed_data(df)
    
    # Step 4: Train models
    results = train_models(df)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED!")
    print("="*60)
    print(f"\nTotal planets processed: {len(df)}")
    print(f"Best model: {results['best_model']} (Accuracy: {results['best_accuracy']:.4f})")
    print("\nNext steps:")
    print("1. Restart Flask server: python app.py")
    print("2. Open browser: http://localhost:5000")
    print("="*60)


if __name__ == "__main__":
    main()
