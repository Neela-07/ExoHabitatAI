# Habitability Score Index (HSI) - Detailed Documentation

## Overview

The **Habitability Score Index (HSI)** is a composite metric that quantifies the potential habitability of an exoplanet based on key planetary parameters. The HSI ranges from 0 to 1, where:
- **0.0 - 0.25**: Non-Habitable
- **0.25 - 0.50**: Low Habitability
- **0.50 - 0.75**: Medium Habitability
- **0.75 - 1.0**: High Habitability

## Formula

```
HSI = 0.3 × Temperature_Factor + 
      0.25 × Radius_Factor + 
      0.25 × Density_Factor + 
      0.2 × Distance_Factor
```

## Components

### 1. Temperature Factor (Weight: 30%)

**Optimal Temperature**: 288 K (Earth's average surface temperature)

**Calculation**:
```
Temperature_Factor = 1 - |Surface_Temp - 288| / 200
```

**Range**: ±200 K from optimal
- At 288 K: Factor = 1.0 (optimal)
- At 88 K or 488 K: Factor = 0.0 (extreme)
- Clipped to [0, 1]

**Reasoning**: Liquid water is most stable around Earth's temperature range (273-323 K).

---

### 2. Radius Factor (Weight: 25%)

**Optimal Radius**: 1.0 R_Earth (Earth radius)

**Calculation**:
```
Radius_Factor = 1 - |Radius - 1.0| / 2
```

**Range**: ±2 R_Earth from optimal
- At 1.0 R_Earth: Factor = 1.0 (optimal)
- At 0 R_Earth or ≥3 R_Earth: Factor = 0.0 (extreme)
- Clipped to [0, 1]

**Reasoning**: 
- Too small (<0.8 R_Earth): Cannot retain atmosphere
- Too large (>1.5 R_Earth): Becomes gas giant or has excessive gravity
- Earth-like radius (0.8-1.5 R_Earth) is ideal for habitability

---

### 3. Density Factor (Weight: 25%)

**Optimal Density**: 4-6 g/cm³ (rocky planet range)

**Calculation**:
```
If 4 ≤ Density ≤ 6:
    Density_Factor = 1.0
Else:
    Density_Factor = 1 - |Density - 5| / 5
```

**Range**: Clipped to [0, 1]

**Reasoning**:
- **4-6 g/cm³**: Rocky/terrestrial planets (like Earth at ~5.5 g/cm³)
- **<4 g/cm³**: Too light, likely ice or gas
- **>6 g/cm³**: Too dense, likely metallic or very compressed

---

### 4. Distance Factor (Weight: 20%)

**Habitable Zone Calculation**:
```
Inner_Boundary = √(Star_Luminosity / 1.1)
Outer_Boundary = √(Star_Luminosity / 0.53)
```

**Calculation**:
```
If Distance within [Inner_Boundary, Outer_Boundary]:
    Distance_Factor = 1.0
Else:
    Distance_Factor = 0.5  (penalty for being outside HZ)
```

**Reasoning**:
- **Within Habitable Zone**: Allows liquid water to exist
- **Too close**: Planet too hot (water boils)
- **Too far**: Planet too cold (water freezes)
- The habitable zone varies based on stellar luminosity

---

## Example Calculation

### Example Planet: Earth-like Planet

**Input Parameters**:
- Surface Temperature: 288 K
- Radius: 1.0 R_Earth
- Density: 5.5 g/cm³
- Distance from Star: 1.0 AU
- Star Luminosity: 1.0 L_Sun

**Calculations**:

1. **Temperature Factor**:
   ```
   = 1 - |288 - 288| / 200
   = 1 - 0 / 200
   = 1.0
   ```

2. **Radius Factor**:
   ```
   = 1 - |1.0 - 1.0| / 2
   = 1 - 0 / 2
   = 1.0
   ```

3. **Density Factor**:
   ```
   = 1.0 (since 4 ≤ 5.5 ≤ 6)
   ```

4. **Distance Factor**:
   ```
   Inner_Boundary = √(1.0 / 1.1) = 0.953 AU
   Outer_Boundary = √(1.0 / 0.53) = 1.373 AU
   
   Since 1.0 AU is within [0.953, 1.373]:
   Distance_Factor = 1.0
   ```

**Final HSI**:
```
HSI = 0.3 × 1.0 + 0.25 × 1.0 + 0.25 × 1.0 + 0.2 × 1.0
    = 0.3 + 0.25 + 0.25 + 0.2
    = 1.0
```

**Result**: HSI = 1.0 (Perfect Habitability)

---

## Code Implementation

The HSI is implemented in `preprocessing/feature_engineering.py`:

```python
def create_habitability_score_index(self, df):
    """
    Create Habitability Score Index (HSI) based on key planetary parameters
    """
    # Temperature factor (optimal around 288K)
    optimal_temp = 288
    temp_factor = 1 - np.abs(df['surface_temp'] - optimal_temp) / 200
    temp_factor = np.clip(temp_factor, 0, 1)
    
    # Radius factor (optimal around 1 Earth radius)
    optimal_radius = 1.0
    radius_factor = 1 - np.abs(df['radius'] - optimal_radius) / 2
    radius_factor = np.clip(radius_factor, 0, 1)
    
    # Density factor (optimal for rocky planets: 4-6 g/cm³)
    optimal_density_min, optimal_density_max = 4, 6
    density_factor = np.where(
        (df['density'] >= optimal_density_min) & (df['density'] <= optimal_density_max),
        1.0,
        1 - np.abs(df['density'] - 5) / 5
    )
    density_factor = np.clip(density_factor, 0, 1)
    
    # Distance factor (habitable zone)
    habitable_zone_inner = np.sqrt(df['star_luminosity'] / 1.1)
    habitable_zone_outer = np.sqrt(df['star_luminosity'] / 0.53)
    
    distance_factor = np.where(
        (df['distance_from_star'] >= habitable_zone_inner) & 
        (df['distance_from_star'] <= habitable_zone_outer),
        1.0,
        0.5
    )
    
    # Calculate HSI as weighted average
    df['habitability_score_index'] = (
        0.3 * temp_factor +
        0.25 * radius_factor +
        0.25 * density_factor +
        0.2 * distance_factor
    )
    
    return df
```

## Usage

The HSI is automatically calculated during feature engineering:

```bash
python preprocessing/feature_engineering.py
```

After processing, the `habitability_score_index` column is added to the dataset and can be used for:
- **ML Model Training**: As a feature or target variable
- **Ranking Exoplanets**: Sort planets by HSI value
- **Visualization**: Create plots showing HSI distribution
- **Analysis**: Identify most habitable exoplanets

## Interpretation

| HSI Range | Classification | Interpretation |
|-----------|---------------|----------------|
| 0.75 - 1.0 | High | Excellent conditions for life as we know it |
| 0.50 - 0.75 | Medium | Potentially habitable with some limitations |
| 0.25 - 0.50 | Low | Marginal conditions, unlikely but possible |
| 0.0 - 0.25 | Non-Habitable | Conditions unsuitable for life |

## Limitations

1. **Simplified Model**: HSI is a simplified index; real habitability depends on many more factors
2. **Earth-Centric**: Based on Earth-like conditions; life might exist in other forms
3. **Missing Factors**: Does not account for:
   - Atmospheric composition
   - Magnetic field
   - Tidal locking
   - Geological activity
   - Presence of moons

## Related Metrics

- **Stellar Compatibility Index (SCI)**: Measures host star influence
- **Combined Habitability Score**: HSI (60%) + SCI (40%)

## References

- Based on principles from NASA's Habitable Zone calculations
- Inspired by research on planetary habitability
- Earth parameters used as reference values

---

**Location in Codebase**: `preprocessing/feature_engineering.py` → `create_habitability_score_index()`

**Output Column Name**: `habitability_score_index`

**Range**: 0.0 to 1.0 (continuous)

