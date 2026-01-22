"""
Create optimized sample dataset from full exoplanet data
"""
import pandas as pd
import numpy as np

# Load full dataset
print('Loading dataset...')
df = pd.read_csv('f:/merged_exoplanet_dataset (1).csv', low_memory=False)
print(f'Loaded {len(df)} rows')

# Get unique planets only
df = df.drop_duplicates(subset=['pl_name'], keep='first')
print(f'Unique planets: {len(df)}')

# Select essential columns
cols = ['pl_name', 'hostname', 'pl_orbper', 'pl_orbsmax', 'pl_rade', 'pl_bmasse', 
        'pl_eqt', 'st_spectype', 'st_teff', 'st_met', 'density', 'luminosity',
        'discoverymethod', 'disc_year', 'sy_dist']

# Keep only columns that exist
existing_cols = [c for c in cols if c in df.columns]
df_small = df[existing_cols].copy()

# Remove rows with no planet name
df_small = df_small.dropna(subset=['pl_name'])

# Rename columns for consistency
rename_map = {
    'pl_orbper': 'orbital_period',
    'pl_orbsmax': 'distance_from_star', 
    'pl_rade': 'radius',
    'pl_bmasse': 'mass',
    'pl_eqt': 'surface_temp',
    'st_spectype': 'star_type',
    'st_teff': 'star_temp',
    'st_met': 'metallicity',
    'luminosity': 'star_luminosity',
    'sy_dist': 'distance_ly'
}
df_small = df_small.rename(columns=rename_map)

# Extract first letter of star type
df_small['star_type'] = df_small['star_type'].astype(str).str[0].str.upper()
df_small.loc[~df_small['star_type'].isin(['O', 'B', 'A', 'F', 'G', 'K', 'M']), 'star_type'] = 'G'

# Calculate habitability score
def calc_score(row):
    score = 0.3  # base score
    
    # Temperature factor (optimal 250-350K for liquid water)
    if pd.notna(row.get('surface_temp')):
        temp = row['surface_temp']
        temp_factor = 1 - min(abs(temp - 300) / 200, 1)
        score += 0.25 * temp_factor
    
    # Size factor (Earth-like 0.8-2 Earth radii optimal)
    if pd.notna(row.get('radius')):
        r = row['radius']
        size_factor = 1 - min(abs(r - 1.2) / 2, 1)
        score += 0.2 * size_factor
    
    # Distance factor (habitable zone ~0.5-2 AU)
    if pd.notna(row.get('distance_from_star')):
        d = row['distance_from_star']
        dist_factor = 1 - min(abs(d - 1.0) / 1.5, 1)
        score += 0.15 * dist_factor
    
    # Star type factor
    star_type = row.get('star_type', 'G')
    star_factors = {'G': 1.0, 'K': 0.9, 'F': 0.8, 'M': 0.6, 'A': 0.4, 'B': 0.2, 'O': 0.1}
    score += 0.1 * star_factors.get(star_type, 0.5)
    
    return max(0, min(1, score))

print('Calculating habitability scores...')
df_small['habitability_score'] = df_small.apply(calc_score, axis=1)
df_small['habitability_class'] = df_small['habitability_score'].apply(
    lambda x: 'High' if x > 0.7 else ('Medium' if x > 0.4 else 'Low'))

# Sort by score (keep all unique planets)
df_small = df_small.sort_values('habitability_score', ascending=False)
df_small['rank'] = range(1, len(df_small) + 1)

# Save to data folder
output_path = 'data/exoplanets_sample.csv'
df_small.to_csv(output_path, index=False)

import os
file_size = os.path.getsize(output_path) / 1024 / 1024
print(f'Saved {len(df_small)} exoplanets to {output_path}')
print(f'File size: {file_size:.2f} MB')

# Show top 10
print('\nTop 10 Most Habitable Exoplanets:')
print(df_small[['rank', 'pl_name', 'habitability_score', 'habitability_class']].head(10).to_string())
