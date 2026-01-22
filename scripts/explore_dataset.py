import pandas as pd
import numpy as np

# Read the dataset
df = pd.read_csv('data/merged_exoplanet_dataset (1).csv', low_memory=False)

print('=' * 80)
print('DETAILED EXOPLANET DATASET EXPLORATION')
print('=' * 80)

print(f'Dataset Dimensions: {df.shape[0]:,} rows × {df.shape[1]} columns')
print(f'Memory Usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB')

# Data completeness analysis
print('\n' + '=' * 80)
print('DATA COMPLETENESS ANALYSIS')
print('=' * 80)

missing_data = df.isnull().sum()
missing_percent = (missing_data / len(df)) * 100

print('Top 20 columns by missing data:')
missing_summary = pd.DataFrame({
    'Missing Count': missing_data,
    'Missing %': missing_percent
}).sort_values('Missing Count', ascending=False).head(20)

for idx, (col, row) in enumerate(missing_summary.iterrows()):
    missing_count = int(row['Missing Count'])
    missing_pct = row['Missing %']
    print(f'{col:<25} {missing_count:>8,} ({missing_pct:>5.1f}%)')

# Key planetary parameters completeness
key_params = ['pl_rade', 'pl_radj', 'pl_bmasse', 'pl_bmassj', 'pl_eqt', 'pl_orbper', 'pl_orbsmax']
print('\nKey Planetary Parameters Completeness:')
for param in key_params:
    if param in df.columns:
        missing = df[param].isnull().sum()
        pct = (missing / len(df)) * 100
        complete = len(df) - missing
        complete_pct = 100 - pct
        print(f'{param:<12} {complete:>6,} complete ({complete_pct:>5.1f}% complete)')

# Data types distribution
print('\n' + '=' * 80)
print('DATA TYPES DISTRIBUTION')
print('=' * 80)

dtype_counts = df.dtypes.value_counts()
for dtype, count in dtype_counts.items():
    print(f'{dtype}: {count} columns')

# Numeric columns analysis
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(f'\nNumeric columns: {len(numeric_cols)}')

# Categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
print(f'Categorical columns: {len(categorical_cols)}')

print('\n' + '=' * 80)
print('STATISTICAL SUMMARY OF KEY PARAMETERS')
print('=' * 80)

key_numeric_cols = ['pl_rade', 'pl_bmasse', 'pl_eqt', 'pl_orbper', 'pl_orbsmax', 'st_teff', 'st_mass', 'st_rad']
available_cols = [col for col in key_numeric_cols if col in df.columns]

summary_stats = df[available_cols].describe()
print(summary_stats.round(2))

print('\n' + '=' * 80)
print('EXPLORING EXTREME VALUES')
print('=' * 80)

# Most massive planets
if 'pl_bmassj' in df.columns:
    massive_planets = df.nlargest(5, 'pl_bmassj')[['pl_name', 'hostname', 'pl_bmassj', 'pl_radj']]
    print('\nMost Massive Planets (Jupiter masses):')
    for _, planet in massive_planets.iterrows():
        name = planet['pl_name']
        host = planet['hostname']
        mass = planet['pl_bmassj']
        print(f'{name:<20} {host:<15} {mass:>8.2f} Mj')

# Hottest planets
if 'pl_eqt' in df.columns:
    hot_planets = df.nlargest(5, 'pl_eqt')[['pl_name', 'hostname', 'pl_eqt']]
    print('\nHottest Planets (K):')
    for _, planet in hot_planets.iterrows():
        name = planet['pl_name']
        host = planet['hostname']
        temp = planet['pl_eqt']
        print(f'{name:<20} {host:<15} {temp:>8.0f} K')

# Largest planets
if 'pl_radj' in df.columns:
    large_planets = df.nlargest(5, 'pl_radj')[['pl_name', 'hostname', 'pl_radj']]
    print('\nLargest Planets (Jupiter radii):')
    for _, planet in large_planets.iterrows():
        name = planet['pl_name']
        host = planet['hostname']
        radius = planet['pl_radj']
        print(f'{name:<20} {host:<15} {radius:>8.2f} Rj')

print('\n' + '=' * 80)
print('DISCOVERY TRENDS OVER TIME')
print('=' * 80)

# Discovery trends
if 'disc_year' in df.columns:
    yearly_discoveries = df['disc_year'].value_counts().sort_index()
    print('Annual Discovery Counts (last 10 years):')
    for year, count in yearly_discoveries.tail(10).items():
        print(f'{int(year)}: {count:,} discoveries')

print('\n' + '=' * 80)
print('CORRELATION ANALYSIS')
print('=' * 80)

# Correlation between key parameters
correlation_cols = ['pl_rade', 'pl_bmasse', 'pl_eqt', 'pl_orbper', 'pl_orbsmax', 'st_teff', 'st_mass']
available_corr_cols = [col for col in correlation_cols if col in df.columns]

if len(available_corr_cols) > 1:
    corr_matrix = df[available_corr_cols].corr()
    print('Correlation Matrix (key parameters):')
    print(corr_matrix.round(3))

print('\n' + '=' * 80)
print('DATA QUALITY ISSUES')
print('=' * 80)

# Check for duplicates
duplicates = df.duplicated().sum()
print(f'Duplicate rows: {duplicates}')

# Check for negative values in parameters that shouldn't be negative
negative_checks = {
    'pl_rade': 'Planet radius',
    'pl_bmasse': 'Planet mass (Earth)',
    'pl_eqt': 'Equilibrium temperature',
    'st_teff': 'Star temperature'
}

for col, desc in negative_checks.items():
    if col in df.columns:
        negatives = (df[col] < 0).sum()
        if negatives > 0:
            print(f'{desc}: {negatives} negative values')

# Check for unrealistic values
if 'pl_eqt' in df.columns:
    extreme_temp = (df['pl_eqt'] > 10000).sum()
    if extreme_temp > 0:
        print(f'Unrealistic temperatures (>10,000K): {extreme_temp}')

print('\n' + '=' * 80)
print('SAMPLE OF INTERESTING SYSTEMS')
print('=' * 80)

# Multi-planet systems
multi_planet = df.groupby('hostname').size().sort_values(ascending=False)
print('Largest multi-planet systems:')
for system, count in multi_planet.head(5).items():
    print(f'{system}: {count} planets')

# Systems with potentially habitable planets
hab_criteria = (
    (df['pl_rade'].between(0.8, 2.0)) &  # Reasonable size
    (df['pl_eqt'].between(200, 350)) &   # Habitable temperature
    (df['pl_bmasse'].between(0.5, 5))    # Reasonable mass
)

hab_candidates = df[hab_criteria]
if len(hab_candidates) > 0:
    print(f'\nPotentially habitable planets found: {len(hab_candidates)}')
    hab_systems = hab_candidates['hostname'].value_counts().head(5)
    print('Systems with most habitable candidates:')
    for system, count in hab_systems.items():
        print(f'{system}: {count} potentially habitable planets')

print('\n' + '=' * 80)
print('UNIQUE INSIGHTS FROM THE DATA')
print('=' * 80)

# Interesting patterns
print('1. Planet Detection Methods:')
methods = df['discoverymethod'].value_counts()
for method, count in methods.head(3).items():
    pct = (count / len(df)) * 100
    print(f'   {method}: {count:,} ({pct:.1f}%)')

print('\n2. Star-Planet Correlations:')
if 'st_teff' in df.columns and 'pl_eqt' in df.columns:
    star_temp_corr = df[['st_teff', 'pl_eqt']].corr().iloc[0, 1]
    print(f'   Star temperature vs Planet temperature correlation: {star_temp_corr:.3f}')

print('\n3. Orbital Period Distribution:')
if 'pl_orbper' in df.columns:
    short_period = (df['pl_orbper'] < 10).sum()
    long_period = (df['pl_orbper'] > 365).sum()
    print(f'   Planets with orbital period < 10 days: {short_period:,}')
    print(f'   Planets with orbital period > 1 year: {long_period:,}')

print('\n4. Size-Mass Relationship:')
if 'pl_rade' in df.columns and 'pl_bmasse' in df.columns:
    size_mass_corr = df[['pl_rade', 'pl_bmasse']].corr().iloc[0, 1]
    print(f'   Planet radius vs mass correlation: {size_mass_corr:.3f}')

print('\n5. Recent Discovery Trends:')
if 'disc_year' in df.columns:
    recent_years = df[df['disc_year'] >= 2020]
    recent_count = len(recent_years)
    recent_pct = (recent_count / len(df)) * 100
    print(f'   Planets discovered since 2020: {recent_count:,} ({recent_pct:.1f}%)')

print('\n' + '=' * 80)
print('DATASET SUMMARY')
print('=' * 80)
print(f'• Total confirmed exoplanets: {len(df):,}')
print(f'• Unique planetary systems: {df["hostname"].nunique():,}')
print(f'• Discovery span: {df["disc_year"].min()} - {df["disc_year"].max()}')
print(f'• Most complete parameter: Planet orbital period ({100 - (df["pl_orbper"].isnull().sum() / len(df)) * 100:.1f}% complete)')
print(f'• Most sparse parameter: Planet density ({100 - (df["density"].isnull().sum() / len(df)) * 100:.1f}% complete)')
print(f'• Largest multi-planet system: {multi_planet.index[0]} with {multi_planet.iloc[0]} planets')
print(f'• Potentially habitable candidates: {len(hab_candidates)} planets meeting basic criteria')