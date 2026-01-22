import pandas as pd
import numpy as np
import requests
import json
import time

# Read the dataset
df = pd.read_csv('data/merged_exoplanet_dataset (1).csv', low_memory=False)

print('=' * 80)
print('EXOPLANET HABITABILITY PREDICTIONS & RANKINGS')
print('=' * 80)

# Filter for planets with sufficient data for prediction
prediction_candidates = df[
    df['pl_rade'].notna() &  # Planet radius
    df['pl_bmasse'].notna() &  # Planet mass
    df['pl_eqt'].notna() &  # Equilibrium temperature
    df['pl_orbper'].notna() &  # Orbital period
    df['pl_orbsmax'].notna() &  # Semi-major axis
    df['st_teff'].notna() &  # Star temperature
    df['st_spectype'].notna()  # Star spectral type
].copy()

print(f'Planets with sufficient data for prediction: {len(prediction_candidates):,}')

# Sample diverse planets for prediction
sample_size = min(50, len(prediction_candidates))  # Limit to 50 for API calls
sample_planets = prediction_candidates.sample(n=sample_size, random_state=42)

print(f'Sampling {sample_size} planets for prediction analysis...')

# Function to convert dataset planet to API format
def convert_to_api_format(planet):
    # Extract star type (first letter)
    star_type = str(planet.get('st_spectype', 'G'))[0].upper()
    if star_type not in ['O', 'B', 'A', 'F', 'G', 'K', 'M']:
        star_type = 'G'  # Default to G-type

    # Convert units to match API expectations
    # API expects: radius (Jupiter radii), mass (Jupiter masses), etc.
    api_data = {
        'radius': planet['pl_radj'] if pd.notna(planet.get('pl_radj')) else planet['pl_rade'] / 11.2,  # Convert Earth to Jupiter radii
        'mass': planet['pl_bmassj'] if pd.notna(planet.get('pl_bmassj')) else planet['pl_bmasse'] / 317.8,  # Convert Earth to Jupiter masses
        'density': planet.get('density', 5.5),  # g/cm³
        'surface_temp': planet['pl_eqt'],  # K
        'orbital_period': planet['pl_orbper'],  # days
        'distance_from_star': planet['pl_orbsmax'],  # AU
        'star_type': star_type,
        'star_luminosity': planet.get('luminosity', 1.0),
        'star_temp': planet['st_teff'],  # K
        'metallicity': planet.get('st_met', 0.0)
    }
    return api_data

# Function to get prediction from API
def get_prediction(api_data):
    try:
        response = requests.post('http://127.0.0.1:5000/api/predict',
                               json=api_data,
                               timeout=10)
        if response.status_code == 200:
            result = response.json()
            if result.get('status') == 'success' and result.get('prediction'):
                prediction = result['prediction']
                return {
                    'class': prediction.get('class', 'Unknown'),
                    'habitability_score': prediction.get('habitability_score', 0),
                    'confidence': prediction.get('probability', 0)
                }
        return None
    except Exception as e:
        print(f'API call failed: {e}')
        return None

# Run predictions on sample planets
print('\nRunning predictions on sample planets...')
predictions = []

for i, (_, planet) in enumerate(sample_planets.iterrows()):
    print(f'Predicting planet {i+1}/{sample_size}: {planet["pl_name"]}...', end='\r')

    api_data = convert_to_api_format(planet)
    prediction = get_prediction(api_data)

    if prediction:
        planet_result = {
            'name': planet['pl_name'],
            'hostname': planet['hostname'],
            'radius_earth': planet.get('pl_rade', np.nan),
            'mass_earth': planet.get('pl_bmasse', np.nan),
            'temperature': planet['pl_eqt'],
            'star_type': str(planet.get('st_spectype', 'Unknown'))[0],
            'prediction_class': prediction['class'],
            'habitability_score': prediction['habitability_score'],
            'confidence': prediction['confidence']
        }
        predictions.append(planet_result)
    else:
        print(f'Failed to predict {planet["pl_name"]}')

print(f'\nSuccessfully predicted {len(predictions)} planets')

# Create DataFrame from predictions
predictions_df = pd.DataFrame(predictions)

# Rank planets by habitability score
ranked_planets = predictions_df.sort_values('habitability_score', ascending=False)

print('\n' + '=' * 80)
print('TOP 20 MOST HABITABLE PLANETS (by AI Prediction)')
print('=' * 80)

for i, (_, planet) in enumerate(ranked_planets.head(20).iterrows(), 1):
    print(f'{i:2d}. {planet["name"]:<20} | {planet["hostname"]:<15} | '
          f'Score: {planet["habitability_score"]:.3f} | '
          f'Class: {planet["prediction_class"]:<8} | '
          f'Temp: {planet["temperature"]:>4.0f}K | '
          f'Confidence: {planet["confidence"]:.1%}')

print('\n' + '=' * 80)
print('HABITABILITY SCORE DISTRIBUTION')
print('=' * 80)

# Analyze prediction distribution
score_ranges = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
score_labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']

predictions_df['score_range'] = pd.cut(predictions_df['habitability_score'],
                                      bins=score_ranges, labels=score_labels, include_lowest=True)

score_distribution = predictions_df['score_range'].value_counts().sort_index()
print('Habitability Score Distribution:')
for range_name, count in score_distribution.items():
    pct = (count / len(predictions_df)) * 100
    print(f'  {range_name}: {count:>2d} planets ({pct:>5.1f}%)')

print('\n' + '=' * 80)
print('CLASS PREDICTION DISTRIBUTION')
print('=' * 80)

class_distribution = predictions_df['prediction_class'].value_counts()
print('Prediction Class Distribution:')
for class_name, count in class_distribution.items():
    pct = (count / len(predictions_df)) * 100
    print(f'  {class_name:<10}: {count:>2d} planets ({pct:>5.1f}%)')

print('\n' + '=' * 80)
print('HIGH-CONFIDENCE HABITABLE CANDIDATES')
print('=' * 80)

# Filter for high-confidence habitable planets
high_confidence_habitable = predictions_df[
    (predictions_df['habitability_score'] > 0.7) &
    (predictions_df['confidence'] > 0.8)
].sort_values('habitability_score', ascending=False)

if len(high_confidence_habitable) > 0:
    print(f'Found {len(high_confidence_habitable)} high-confidence habitable candidates:')
    for i, (_, planet) in enumerate(high_confidence_habitable.iterrows(), 1):
        print(f'{i}. {planet["name"]:<20} | Score: {planet["habitability_score"]:.3f} | '
              f'Confidence: {planet["confidence"]:.1%} | Temp: {planet["temperature"]:>4.0f}K')
else:
    print('No high-confidence habitable candidates found in this sample.')

print('\n' + '=' * 80)
print('TEMPERATURE vs HABITABILITY ANALYSIS')
print('=' * 80)

# Analyze temperature correlation with habitability
temp_corr = predictions_df[['temperature', 'habitability_score']].corr().iloc[0, 1]
print(f'Temperature vs Habitability Score Correlation: {temp_corr:.3f}')

# Group by temperature ranges
temp_ranges = [0, 200, 250, 300, 350, 400, float('inf')]
temp_labels = ['<200K', '200-250K', '250-300K', '300-350K', '350-400K', '>400K']

predictions_df['temp_range'] = pd.cut(predictions_df['temperature'],
                                     bins=temp_ranges, labels=temp_labels, right=False)

temp_habitability = predictions_df.groupby('temp_range')['habitability_score'].agg(['mean', 'count'])
print('\nAverage Habitability Score by Temperature Range:')
for temp_range, stats in temp_habitability.iterrows():
    print(f'  {temp_range:<10}: {stats["mean"]:>5.3f} (n={int(stats["count"])})')

print('\n' + '=' * 80)
print('STAR TYPE IMPACT ON HABITABILITY')
print('=' * 80)

star_habitability = predictions_df.groupby('star_type')['habitability_score'].agg(['mean', 'count', 'std'])
print('Average Habitability Score by Host Star Type:')
for star_type, stats in star_habitability.iterrows():
    print(f'  {star_type} stars: {stats["mean"]:>5.3f} ± {stats["std"]:>5.3f} (n={int(stats["count"])})')

print('\n' + '=' * 80)
print('SUMMARY STATISTICS')
print('=' * 80)

print(f'Total planets analyzed: {len(predictions_df)}')
print(f'Average habitability score: {predictions_df["habitability_score"].mean():.3f}')
print(f'Median habitability score: {predictions_df["habitability_score"].median():.3f}')
print(f'Highest habitability score: {predictions_df["habitability_score"].max():.3f}')
print(f'Lowest habitability score: {predictions_df["habitability_score"].min():.3f}')
print(f'Average prediction confidence: {predictions_df["confidence"].mean():.1%}')

# Save results to CSV
ranked_planets.to_csv('prediction_results.csv', index=False)
print(f'\nResults saved to: prediction_results.csv')