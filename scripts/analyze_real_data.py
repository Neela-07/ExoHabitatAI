"""
Analyze actual dataset to understand available features
"""
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('data/raw/exoplanets_raw.csv')
print(f"Total records: {len(df):,}")
print(f"Total columns: {len(df.columns)}")

# Key columns we need for habitability
key_cols = {
    'pl_rade': 'Planet Radius (Earth radii)',
    'pl_bmasse': 'Planet Mass (Earth masses)',
    'pl_orbper': 'Orbital Period (days)',
    'pl_orbsmax': 'Semi-Major Axis (AU)',
    'pl_eqt': 'Equilibrium Temperature (K)',
    'st_teff': 'Stellar Temperature (K)',
    'st_rad': 'Stellar Radius (Solar radii)',
    'st_mass': 'Stellar Mass (Solar masses)',
    'st_met': 'Stellar Metallicity',
    'st_logg': 'Stellar Surface Gravity'
}

print("\n" + "="*70)
print("DATA AVAILABILITY ANALYSIS")
print("="*70)

for col, desc in key_cols.items():
    if col in df.columns:
        non_null = df[col].notna().sum()
        pct = (non_null/len(df))*100
        print(f"{desc:40} {col:15} {non_null:8,} ({pct:5.1f}%)")
        
        if non_null > 0:
            print(f"  Range: {df[col].min():.3f} to {df[col].max():.3f}")
            print(f"  Median: {df[col].median():.3f}")
    else:
        print(f"{desc:40} {col:15} NOT FOUND")

print("\n" + "="*70)
print("SAMPLE DATA (first 5 rows with most complete data)")
print("="*70)

# Find rows with most complete data
available_cols = [c for c in key_cols.keys() if c in df.columns]
completeness = df[available_cols].notna().sum(axis=1)
best_rows = df.loc[completeness.nlargest(5).index]

for col in available_cols:
    print(f"\n{col}:")
    print(best_rows[col].values)
