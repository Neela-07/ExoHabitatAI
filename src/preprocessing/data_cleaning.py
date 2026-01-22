"""
Module 2: Data Cleaning
Handles missing values, outliers, and inconsistent entries
Validates data quality using descriptive statistics and visualization
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config import DATABASE_CONFIG, PROCESSED_DATA_DIR
from src.utils.database import DatabaseManager

class DataCleaner:
    """
    Cleans and preprocesses exoplanet data
    """
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        
    def clean_data(self, df):
        """
        Main cleaning pipeline
        """
        print("Starting data cleaning process...")
        original_shape = df.shape
        print(f"Original dataset shape: {original_shape}")
        
        # Step 1: Handle duplicates
        df = self.remove_duplicates(df)
        
        # Step 2: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 3: Handle outliers
        df = self.handle_outliers(df)
        
        # Step 4: Fix inconsistent entries
        df = self.fix_inconsistent_entries(df)
        
        # Step 5: Validate data types
        df = self.validate_data_types(df)
        
        final_shape = df.shape
        print(f"Final dataset shape: {final_shape}")
        print(f"Removed {original_shape[0] - final_shape[0]} rows, {original_shape[1] - final_shape[1]} columns")
        
        return df
    
    def remove_duplicates(self, df):
        """
        Remove duplicate rows
        """
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            print(f"Removing {duplicates} duplicate rows...")
            df = df.drop_duplicates().reset_index(drop=True)
        return df
    
    def handle_missing_values(self, df):
        """
        Handle missing values using appropriate strategies
        """
        print("\nHandling missing values...")
        
        # Map NASA columns to our internal names
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
            'st_logg': 'star_logg'
        }
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Numerical columns: fill with median
        numerical_cols = ['radius', 'mass', 'surface_temp', 
                         'orbital_period', 'distance_from_star', 
                         'star_radius', 'star_temp', 'star_mass', 'metallicity', 'star_logg']
        
        for col in numerical_cols:
            if col in df.columns:
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    median_value = df[col].median()
                    df[col].fillna(median_value, inplace=True)
                    print(f"  {col}: Filled {missing_count} missing values with median {median_value:.2f}")
        
        # Extract star type from spectral type (st_spectype)
        if 'st_spectype' in df.columns:
            def extract_star_type(spectype):
                if pd.isna(spectype):
                    return 'G'  # Default to G-type
                spec_str = str(spectype).strip().upper()
                if len(spec_str) > 0 and spec_str[0] in ['O', 'B', 'A', 'F', 'G', 'K', 'M']:
                    return spec_str[0]
                return 'G'
            
            df['star_type'] = df['st_spectype'].apply(extract_star_type)
            print(f"  star_type: Extracted from st_spectype")
        else:
            df['star_type'] = 'G'
            print(f"  star_type: Set default to 'G'")
        
        # Remove rows with too many missing values (if any remain)
        threshold = len(df.columns) * 0.5  # Remove if more than 50% missing
        df = df.dropna(thresh=threshold)
        
        return df
    
    def handle_outliers(self, df):
        """
        Handle outliers using IQR method for numerical columns
        """
        print("\nHandling outliers using IQR method...")
        
        numerical_cols = ['radius', 'mass', 'density', 'surface_temp', 
                         'orbital_period', 'distance_from_star', 
                         'star_luminosity', 'star_temp', 'metallicity']
        
        outlier_count = 0
        for col in numerical_cols:
            if col in df.columns and df[col].dtype in [np.int64, np.float64]:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                if outliers > 0:
                    # Cap outliers instead of removing
                    df.loc[df[col] < lower_bound, col] = lower_bound
                    df.loc[df[col] > upper_bound, col] = upper_bound
                    print(f"  {col}: Capped {outliers} outliers")
                    outlier_count += outliers
        
        return df
    
    def fix_inconsistent_entries(self, df):
        """
        Fix inconsistent entries
        """
        print("\nFixing inconsistent entries...")
        
        # Ensure star_type is uppercase
        if 'star_type' in df.columns:
            df['star_type'] = df['star_type'].astype(str).str.upper().str[0]
            valid_types = ['O', 'B', 'A', 'F', 'G', 'K', 'M']
            df['star_type'] = df['star_type'].apply(
                lambda x: x if x in valid_types else 'G'
            )
        
        # Ensure positive values for physical quantities
        positive_cols = ['radius', 'mass', 'density', 'surface_temp', 
                        'orbital_period', 'distance_from_star', 
                        'star_luminosity', 'star_temp']
        for col in positive_cols:
            if col in df.columns:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    df[col] = df[col].abs()
                    print(f"  {col}: Fixed {negative_count} negative values")
        
        return df
    
    def validate_data_types(self, df):
        """
        Validate and convert data types
        """
        print("\nValidating data types...")
        
        # Convert numerical columns
        numerical_cols = ['radius', 'mass', 'density', 'surface_temp', 
                         'orbital_period', 'distance_from_star', 
                         'star_luminosity', 'star_temp', 'metallicity']
        
        for col in numerical_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Ensure categorical columns are strings
        if 'star_type' in df.columns:
            df['star_type'] = df['star_type'].astype(str)
        
        return df
    
    def generate_descriptive_statistics(self, df):
        """
        Generate descriptive statistics for data quality validation
        """
        print("\n" + "="*60)
        print("DESCRIPTIVE STATISTICS")
        print("="*60)
        print(df.describe())
        print("\nData types:")
        print(df.dtypes)
        print(f"\nTotal records: {len(df)}")
        print(f"Total features: {len(df.columns)}")
        return df.describe()

def main():
    """
    Main function for data cleaning
    """
    cleaner = DataCleaner()
    
    # Load raw data
    df = cleaner.db_manager.load_data(source="raw")
    
    if df is None:
        print("Error: Could not load raw data. Please run data collection first.")
        return
    
    # Clean data
    cleaned_df = cleaner.clean_data(df)
    
    # Generate statistics
    cleaner.generate_descriptive_statistics(cleaned_df)
    
    # Save cleaned data
    cleaner.db_manager.save_data(cleaned_df, destination="processed")
    
    print("\nData cleaning completed successfully!")

if __name__ == "__main__":
    main()
