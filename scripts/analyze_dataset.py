import pandas as pd
import numpy as np

# Read the merged dataset
df = pd.read_csv('data/merged_exoplanet_dataset (1).csv', low_memory=False)

print('=' * 60)
print('EXOPLANET DATASET ANALYSIS - KEY OUTPUTS')
print('=' * 60)

# Basic statistics
total_planets = len(df)
unique_stars = df['hostname'].nunique()
min_year = df['disc_year'].min()
max_year = df['disc_year'].max()

print(f'Total exoplanets: {total_planets:,}')
print(f'Unique host stars: {unique_stars:,}')
print(f'Discovery years range: {min_year} - {max_year}')

# Most common discovery methods
print('\nTop 5 Discovery Methods:')
discovery_methods = df['discoverymethod'].value_counts().head(5)
for method, count in discovery_methods.items():
    print(f'  {method}: {count:,} planets')

print('\n' + '=' * 60)
print('PLANET SIZE DISTRIBUTION')
print('=' * 60)

# Planet size distribution (Earth radii)
size_bins = [0, 1.25, 2, 4, 6, 11.2, float('inf')]
size_labels = ['Earth-sized (<1.25)', 'Super-Earth (1.25-2)', 'Sub-Neptune (2-4)',
               'Sub-Saturn (4-6)', 'Jupiter-sized (6-11.2)', 'Gas Giant (>11.2)']
df['size_category'] = pd.cut(df['pl_rade'], bins=size_bins, labels=size_labels, right=False)
size_dist = df['size_category'].value_counts()
for category, count in size_dist.items():
    if pd.notna(category):
        print(f'  {category}: {count:,} planets')

print('\n' + '=' * 60)
print('TEMPERATURE ANALYSIS')
print('=' * 60)

# Temperature analysis
temp_data = df['pl_eqt'].dropna()
if len(temp_data) > 0:
    print(f'Planets with temperature data: {len(temp_data):,}')
    print(f'Hottest planet: {temp_data.max():.0f} K')
    print(f'Coldest planet: {temp_data.min():.0f} K')
    print(f'Median temperature: {temp_data.median():.0f} K')

    # Habitability zone candidates (rough estimate: 200-400K)
    habitable_temp = temp_data[(temp_data >= 200) & (temp_data <= 400)]
    print(f'Planets in habitable temperature range (200-400K): {len(habitable_temp):,}')

print('\n' + '=' * 60)
print('HOST STAR ANALYSIS')
print('=' * 60)

# Star types
star_types = df['st_spectype'].dropna().str[0].value_counts().head(10)
print('Top 10 Host Star Spectral Types:')
for star_type, count in star_types.items():
    print(f'  Type {star_type}: {count:,} stars')

print('\n' + '=' * 60)
print('POTENTIAL HABITABLE CANDIDATES')
print('=' * 60)

# Filter for potentially habitable planets
habitable_candidates = df[
    (df['pl_rade'].between(0.5, 2.5)) &  # Earth-sized to Super-Earth
    (df['pl_eqt'].between(200, 400)) &   # Habitable temperature
    (df['pl_bmasse'].between(0.5, 10))   # Reasonable mass
].copy()

print(f'Number of potentially habitable candidates: {len(habitable_candidates):,}')

if len(habitable_candidates) > 0:
    print('\nTop 10 Most Promising Candidates (by temperature similarity to Earth):')
    # Sort by temperature (closer to Earth's 255K is better)
    habitable_candidates['temp_diff'] = abs(habitable_candidates['pl_eqt'] - 255)
    top_candidates = habitable_candidates.nsmallest(10, 'temp_diff')

    for i, (_, planet) in enumerate(top_candidates.iterrows(), 1):
        print(f'{i}. {planet["pl_name"]}')
        print(f'   Host Star: {planet["hostname"]}')
        print(f'   Radius: {planet["pl_rade"]:.2f} Earth radii')
        print(f'   Mass: {planet["pl_bmasse"]:.2f} Earth masses')
        print(f'   Temperature: {planet["pl_eqt"]:.0f} K')
        distance = planet.get('pl_orbsmax')
        if pd.notna(distance):
            print(f'   Distance from star: {distance:.3f} AU')
        print()

print('\n' + '=' * 60)
print('SAMPLE PREDICTIONS USING YOUR MODEL')
print('=' * 60)

# Test predictions on a few sample planets
from app import app
import json

sample_planets = [
    {
        'radius': 1.0, 'mass': 1.0, 'density': 5.5, 'surface_temp': 255,
        'orbital_period': 365, 'distance_from_star': 1.0, 'star_type': 'G',
        'star_luminosity': 1.0, 'star_temp': 5778, 'metallicity': 0.0
    },
    {
        'radius': 1.5, 'mass': 3.0, 'density': 8.0, 'surface_temp': 280,
        'orbital_period': 200, 'distance_from_star': 0.8, 'star_type': 'K',
        'star_luminosity': 0.3, 'star_temp': 4500, 'metallicity': -0.2
    }
]

with app.test_client() as client:
    for i, planet_data in enumerate(sample_planets, 1):
        response = client.post('/api/predict',
                              data=json.dumps(planet_data),
                              content_type='application/json')
        if response.status_code == 200:
            result = json.loads(response.data)
            prediction = result.get('prediction', {})
            print(f'Sample {i} Prediction:')
            print(f'  Class: {prediction.get("class", "Unknown")}')
            print(f'  Habitability Score: {prediction.get("habitability_score", 0):.3f}')
            print(f'  Confidence: {prediction.get("probability", 0):.3f}')
            print()