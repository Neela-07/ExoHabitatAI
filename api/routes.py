"""
Module 5: Flask Backend API
REST APIs to:
- Accept exoplanet data input from users
- Return habitability prediction and ranking
- Connect Flask backend to the database for dynamic data retrieval
- Secure endpoints and implement JSON response structure
"""
from flask import Blueprint, request, jsonify, send_file
import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path
from io import BytesIO
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))
from config import MODELS_DIR, MODEL_CONFIG
from src.utils.database import DatabaseManager

api_bp = Blueprint('api', __name__, url_prefix='/api')

db_manager = DatabaseManager()

def load_model(model_name="best"):
    """
    Load ML model from file
    """
    try:
        if model_name == "best":
            model_path = MODELS_DIR / "best_model.pkl"
        elif model_name == "random_forest":
            model_path = Path(MODEL_CONFIG["random_forest"]["path"])
        elif model_name == "xgboost":
            model_path = Path(MODEL_CONFIG["xgboost"]["path"])
        else:
            model_path = MODELS_DIR / f"{model_name}_model.pkl"
        
        if not model_path.exists():
            return None, f"Model file not found: {model_path}"
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        return model, None
    except Exception as e:
        return None, str(e)

def load_scaler():
    """
    Load feature scaler
    """
    try:
        scaler_path = MODELS_DIR / "scaler.pkl"
        if not scaler_path.exists():
            return None, "Scaler file not found"
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        return scaler, None
    except Exception as e:
        return None, str(e)

def load_feature_columns():
    """
    Load saved feature column names
    """
    try:
        features_path = MODELS_DIR / "feature_columns.pkl"
        if not features_path.exists():
            return None, "Feature columns file not found"
        
        with open(features_path, 'rb') as f:
            feature_columns = pickle.load(f)
        
        return feature_columns, None
    except Exception as e:
        return None, str(e)

def load_label_encoder():
    """
    Load label encoder (if available)
    """
    try:
        encoder_path = MODELS_DIR / "label_encoder.pkl"
        if not encoder_path.exists():
            return None, None  # Not an error if encoder doesn't exist
        
        with open(encoder_path, 'rb') as f:
            encoder = pickle.load(f)
        
        return encoder, None
    except Exception as e:
        return None, None

def calculate_habitability_score(input_data):
    """
    Calculate habitability score from input data
    Note: Input data may already be in Earth units from the form
    """
    try:
        # Extract values
        surface_temp = float(input_data.get('surface_temp', input_data.get('temp', 300)))
        star_temp = float(input_data.get('star_temp', input_data.get('star_temperature', 5772)))
        distance = float(input_data.get('distance_from_star', input_data.get('distance', 1)))
        radius_input = float(input_data.get('radius', 1))
        mass_input = float(input_data.get('mass', 1))
        density = float(input_data.get('density', 5.5))
        metallicity = float(input_data.get('metallicity', 0))
        
        # Input is in Earth units (from form), but prepare_features converts Jupiter to Earth
        # So here we use the input directly assuming Earth units
        # If input was Jupiter units, convert: radius_earth = radius_input * 11.2
        # For now, assume input is in Earth units as that's what the form uses
        radius_earth = radius_input  # Already in Earth radii from form
        mass_earth = mass_input      # Already in Earth masses from form
        
        # Simple habitability score calculation
        # Temperature factor (optimal 273-373K for liquid water)
        temp_factor = 1.0 - min(abs(surface_temp - 300) / 200, 1.0)
        
        # Distance factor (rough habitable zone calculation)
        hz_distance = (star_temp / 5772) ** 2 * 1.0  # AU (simplified HZ distance)
        distance_factor = 1.0 - min(abs(distance - hz_distance) / hz_distance, 1.0)
        
        # Size factor (Earth-like: 0.5-2.0 Earth radii optimal)
        size_factor = 1.0 - min(abs(radius_earth - 1.0) / 1.5, 1.0)
        
        # Mass factor (Earth-like: 0.5-10 Earth masses optimal)
        mass_factor = 1.0 - min(abs(mass_earth - 1.0) / 5.0, 1.0)
        
        # Metallicity factor (optimal around solar metallicity 0.0)
        metal_factor = 1.0 - min(abs(metallicity) / 1.0, 1.0)
        
        # Combined score (weighted average)
        habitability_score = (
            0.4 * temp_factor +
            0.3 * distance_factor +
            0.15 * size_factor +
            0.1 * mass_factor +
            0.05 * metal_factor
        )
        
        return max(0.0, min(1.0, habitability_score))
        
    except Exception as e:
        return 0.5  # Default score

def prepare_features(input_data):
    """
    Prepare features from input data for prediction
    Uses saved feature columns from training
    """
    # Try to load saved feature columns first
    expected_features, error = load_feature_columns()
    
    if error or expected_features is None:
        # Fallback: try to get from scaler
        scaler, error = load_scaler()
        if error or scaler is None:
            # Fallback to default if scaler not available
            expected_features = ['density']
        else:
            # Get feature names from the trained scaler
            if hasattr(scaler, 'feature_names_in_'):
                expected_features = list(scaler.feature_names_in_)
            else:
                # Fallback if feature names not available
                expected_features = ['density']
    
    # Create feature vector with expected features initialized to default values
    features = {col: 1.99 for col in expected_features}  # Use median density as default
    
    # Simple mapping for the actual features the model expects
    # Map input keys to expected feature names
    for key, value in input_data.items():
        # Try to extract relevant values
        try:
            numeric_value = float(value)
            
            # Map to expected features if they match
            if key == 'density' and 'density' in expected_features:
                features['density'] = numeric_value
            elif key in expected_features:
                features[key] = numeric_value
            # Also handle common aliases
            elif key == 'temp' and 'surface_temp' in expected_features:
                features['surface_temp'] = numeric_value
            elif key == 'period' and 'orbital_period' in expected_features:
                features['orbital_period'] = numeric_value
            elif key == 'distance' and 'distance_from_star' in expected_features:
                features['distance_from_star'] = numeric_value
            elif key == 'luminosity' and 'star_luminosity' in expected_features:
                features['star_luminosity'] = numeric_value
            elif key == 'star_temperature' and 'star_temp' in expected_features:
                features['star_temp'] = numeric_value
                
        except (ValueError, TypeError):
            # If conversion fails, keep default value
            pass
    
    # Calculate derived features if they're expected
    if 'radius' in features and 'distance_from_star' in features:
        if 'radius_distance_ratio' in expected_features:
            if features['distance_from_star'] > 0:
                features['radius_distance_ratio'] = features['radius'] / features['distance_from_star']
    
    if 'mass' in features and 'radius' in features:
        if 'mass_radius_ratio' in expected_features:
            if features['radius'] > 0:
                features['mass_radius_ratio'] = features['mass'] / features['radius']
    
    if 'surface_temp' in features and 'density' in features:
        if 'temp_density_interaction' in expected_features:
            features['temp_density_interaction'] = features['surface_temp'] * features['density']
    
    # Handle star_type one-hot encoding
    star_type = input_data.get('star_type', 'G').upper()
    for st in ['A', 'B', 'F', 'G', 'K', 'M']:
        col_name = f'star_type_{st}'
        if col_name in expected_features:
            features[col_name] = 1.0 if st == star_type else 0.0
    
    # Create DataFrame with features in correct order
    feature_df = pd.DataFrame([features], columns=expected_features)
    
    # Select only the features (no target variables)
    X = feature_df[expected_features]
    
    return X, expected_features

@api_bp.route('/predict', methods=['POST'])
def predict():
    """
    Predict habitability from input exoplanet data
    """
    try:
        # Get input data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No data provided',
                'status': 'error'
            }), 400
        
        # Prepare features
        X, feature_names = prepare_features(data)
        
        if X.empty:
            return jsonify({
                'error': 'Invalid input data',
                'status': 'error'
            }), 400
        
        # Load model and scaler
        model, error = load_model("xgboost")
        if error:
            return jsonify({
                'error': f'Model loading error: {error}',
                'status': 'error'
            }), 500
        
        scaler, error = load_scaler()
        if error:
            return jsonify({
                'error': f'Scaler loading error: {error}',
                'status': 'error'
            }), 500
        
        # Scale features
        if scaler:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X.values
        
        # Predict
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0] if hasattr(model, 'predict_proba') else None
        
        # Decode prediction if encoder available
        encoder, _ = load_label_encoder()
        if encoder and hasattr(encoder, 'inverse_transform'):
            try:
                prediction_label = encoder.inverse_transform([prediction])[0]
            except:
                prediction_label = str(prediction)
        else:
            prediction_label = str(prediction)
        
        # Prepare response - ensure habitability_score is always included
        habitability_score = calculate_habitability_score(data)
        
        # Ensure habitability_score is a valid number
        if habitability_score is None or not isinstance(habitability_score, (int, float)):
            habitability_score = 0.5  # Default score
        
        # Clamp between 0 and 1
        habitability_score = max(0.0, min(1.0, float(habitability_score)))
        
        # Determine classification based on habitability score
        if habitability_score >= 0.75:
            classification = "High Habitability"
        elif habitability_score >= 0.50:
            classification = "Medium Habitability"
        elif habitability_score >= 0.25:
            classification = "Low Habitability"
        else:
            classification = "Non-Habitable"
        
        response = {
            'status': 'success',
            'prediction': {
                'class': classification,
                'model_class': prediction_label,
                'probability': float(probability.max()) if probability is not None else 0.0,
                'probabilities': probability.tolist() if probability is not None else None,
                'habitability_score': habitability_score
            }
        }
        
        print(f"API Response - Score: {habitability_score}, Class: {classification}")  # Debug print
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@api_bp.route('/planets', methods=['GET'])
def get_planets():
    """
    Retrieve exoplanet data from database
    """
    try:
        # Load data
        df = db_manager.load_data(source="processed")
        
        if df is None:
            return jsonify({
                'error': 'No data available',
                'status': 'error'
            }), 404
        
        # Get query parameters
        limit = int(request.args.get('limit', 100))
        offset = int(request.args.get('offset', 0))
        
        # Paginate
        df_paginated = df.iloc[offset:offset+limit]
        
        # Replace NaN with None for valid JSON
        df_paginated = df_paginated.replace({np.nan: None, np.inf: None, -np.inf: None})
        
        # Convert to JSON
        planets = df_paginated.to_dict(orient='records')
        
        response = {
            'status': 'success',
            'data': planets,
            'total': len(df),
            'limit': limit,
            'offset': offset
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@api_bp.route('/rankings', methods=['GET'])
def get_rankings():
    """
    Get habitability rankings of exoplanets
    """
    try:
        # Load data
        df = db_manager.load_data(source="processed")
        
        if df is None:
            return jsonify({
                'error': 'No data available',
                'status': 'error'
            }), 404
        
        # Load model for ranking
        model, error = load_model()
        if error:
            return jsonify({
                'error': f'Model loading error: {error}',
                'status': 'error'
            }), 500
        
        # Use the same feature preparation as prediction endpoint
        # Expected features in exact order (from model training) - no star_type_O
        expected_features = [
            'radius', 'mass', 'density', 'surface_temp', 'orbital_period', 'distance_from_star',
            'star_luminosity', 'star_temp', 'metallicity',
            'radius_distance_ratio', 'mass_radius_ratio', 'temp_density_interaction',
            'star_type_A', 'star_type_B', 'star_type_F', 'star_type_G', 'star_type_K', 'star_type_M'
        ]
        
        # Calculate derived features for all planets
        if 'radius' in df.columns and 'distance_from_star' in df.columns:
            df['radius_distance_ratio'] = df['radius'] / (df['distance_from_star'] + 1e-6)
        else:
            df['radius_distance_ratio'] = 0.0
        
        if 'mass' in df.columns and 'radius' in df.columns:
            df['mass_radius_ratio'] = df['mass'] / (df['radius'] + 1e-6)
        else:
            df['mass_radius_ratio'] = 0.0
        
        if 'surface_temp' in df.columns and 'density' in df.columns:
            df['temp_density_interaction'] = df['surface_temp'] * df['density']
        else:
            df['temp_density_interaction'] = 0.0
        
        # Ensure star_type columns exist (one-hot encoded) - excluding 'O' type as it wasn't in training
        star_types = ['B', 'A', 'F', 'G', 'K', 'M']
        for st in star_types:
            col_name = f'star_type_{st}'
            if col_name not in df.columns:
                if 'star_type' in df.columns:
                    df[col_name] = (df['star_type'] == st).astype(int)
                else:
                    df[col_name] = 0
        
        # Select features that exist
        available_features = [f for f in expected_features if f in df.columns]
        X = df[available_features].fillna(df[available_features].median())
        
        # Ensure all expected features are present
        for feat in expected_features:
            if feat not in X.columns:
                X[feat] = 0.0
        
        # Reorder to match expected_features
        X = X[expected_features]
        
        # Load scaler
        scaler, _ = load_scaler()
        if scaler:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X.values
        
        # Predict
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled) if hasattr(model, 'predict_proba') else None
        
        # Decode predictions if encoder available
        encoder, _ = load_label_encoder()
        if encoder and hasattr(encoder, 'inverse_transform'):
            try:
                predicted_classes = encoder.inverse_transform(predictions)
            except:
                predicted_classes = [str(p) for p in predictions]
        else:
            predicted_classes = [str(p) for p in predictions]
        
        # Add predictions to dataframe
        if probabilities is not None:
            df['predicted_score'] = probabilities.max(axis=1)
        else:
            df['predicted_score'] = [float(p) for p in predictions]
        
        df['predicted_class'] = predicted_classes
        
        # Rank by score
        df_ranked = df.sort_values('predicted_score', ascending=False)
        
        # Get top N
        top_n = int(request.args.get('top', 10))
        df_top = df_ranked.head(top_n)
        
        # Replace NaN with None for valid JSON
        df_top = df_top.replace({np.nan: None, np.inf: None, -np.inf: None})
        
        # Convert to JSON
        rankings = df_top.to_dict(orient='records')
        
        response = {
            'status': 'success',
            'rankings': rankings,
            'total': len(df_ranked)
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@api_bp.route('/statistics', methods=['GET'])
def get_statistics():
    """
    Get dataset statistics
    """
    try:
        # Load data
        df = db_manager.load_data(source="processed")
        
        if df is None:
            return jsonify({
                'error': 'No data available',
                'status': 'error'
            }), 404
        
        # Calculate statistics
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        stats = df[numerical_cols].describe().to_dict()
        
        # Additional statistics
        additional_stats = {
            'total_records': len(df),
            'total_features': len(df.columns),
            'numerical_features': len(numerical_cols),
            'categorical_features': len(df.columns) - len(numerical_cols)
        }
        
        response = {
            'status': 'success',
            'statistics': stats,
            'additional': additional_stats
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@api_bp.route('/export/excel', methods=['GET'])
def export_excel():
    """
    Export top habitable exoplanets to Excel
    """
    try:
        # Load data
        df = db_manager.load_data(source="processed")
        if df is None:
            return jsonify({'error': 'No data available'}), 404
        
        # Get rankings (reuse logic from /rankings endpoint)
        model, error = load_model()
        if error:
            return jsonify({'error': f'Model error: {error}'}), 500
        
        # Prepare features
        expected_features = [
            'radius', 'mass', 'density', 'surface_temp', 'orbital_period', 'distance_from_star',
            'star_luminosity', 'star_temp', 'metallicity',
            'radius_distance_ratio', 'mass_radius_ratio', 'temp_density_interaction',
            'star_type_A', 'star_type_B', 'star_type_F', 'star_type_G', 'star_type_K', 'star_type_M'
        ]
        
        # Calculate derived features
        if 'radius' in df.columns and 'distance_from_star' in df.columns:
            df['radius_distance_ratio'] = df['radius'] / (df['distance_from_star'] + 1e-6)
        if 'mass' in df.columns and 'radius' in df.columns:
            df['mass_radius_ratio'] = df['mass'] / (df['radius'] + 1e-6)
        if 'surface_temp' in df.columns and 'density' in df.columns:
            df['temp_density_interaction'] = df['surface_temp'] * df['density']
        
        # Ensure star_type columns
        for st in ['A', 'B', 'F', 'G', 'K', 'M']:
            col_name = f'star_type_{st}'
            if col_name not in df.columns:
                if 'star_type' in df.columns:
                    df[col_name] = (df['star_type'] == st).astype(int)
                else:
                    df[col_name] = 0
        
        # Select and prepare features
        available_features = [f for f in expected_features if f in df.columns]
        X = df[available_features].fillna(df[available_features].median())
        for feat in expected_features:
            if feat not in X.columns:
                X[feat] = 0.0
        X = X[expected_features]
        
        # Scale and predict
        scaler, _ = load_scaler()
        X_scaled = scaler.transform(X) if scaler else X.values
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled) if hasattr(model, 'predict_proba') else None
        
        # Decode predictions
        encoder, _ = load_label_encoder()
        if encoder and hasattr(encoder, 'inverse_transform'):
            predicted_classes = encoder.inverse_transform(predictions)
        else:
            predicted_classes = [str(p) for p in predictions]
        
        # Add predictions
        df['habitability_score'] = probabilities.max(axis=1) if probabilities is not None else predictions
        df['predicted_class'] = predicted_classes
        df['rank'] = range(1, len(df) + 1)
        
        # Sort by score
        df_export = df.sort_values('habitability_score', ascending=False)
        
        # Get top N
        top_n = int(request.args.get('top', 100))
        df_export = df_export.head(top_n)
        
        # Select columns for export
        export_columns = ['rank', 'pl_name', 'habitability_score', 'predicted_class',
                         'radius', 'mass', 'density', 'surface_temp', 'orbital_period',
                         'distance_from_star', 'star_temp', 'star_type']
        available_export_cols = [c for c in export_columns if c in df_export.columns]
        df_export = df_export[available_export_cols].reset_index(drop=True)
        df_export['rank'] = range(1, len(df_export) + 1)
        
        # Create Excel file in memory
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_export.to_excel(writer, sheet_name='Top Habitable Exoplanets', index=False)
        output.seek(0)
        
        # Generate filename
        filename = f'top_habitable_exoplanets_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
        
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@api_bp.route('/export/pdf', methods=['GET'])
def export_pdf():
    """
    Export top habitable exoplanets to PDF
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        
        # Load data (reuse logic from excel export)
        df = db_manager.load_data(source="processed")
        if df is None:
            return jsonify({'error': 'No data available'}), 404
        
        model, error = load_model()
        if error:
            return jsonify({'error': f'Model error: {error}'}), 500
        
        # Prepare features (same as excel export)
        expected_features = [
            'radius', 'mass', 'density', 'surface_temp', 'orbital_period', 'distance_from_star',
            'star_luminosity', 'star_temp', 'metallicity',
            'radius_distance_ratio', 'mass_radius_ratio', 'temp_density_interaction',
            'star_type_A', 'star_type_B', 'star_type_F', 'star_type_G', 'star_type_K', 'star_type_M'
        ]
        
        # Calculate derived features
        if 'radius' in df.columns and 'distance_from_star' in df.columns:
            df['radius_distance_ratio'] = df['radius'] / (df['distance_from_star'] + 1e-6)
        if 'mass' in df.columns and 'radius' in df.columns:
            df['mass_radius_ratio'] = df['mass'] / (df['radius'] + 1e-6)
        if 'surface_temp' in df.columns and 'density' in df.columns:
            df['temp_density_interaction'] = df['surface_temp'] * df['density']
        
        # Ensure star_type columns
        for st in ['A', 'B', 'F', 'G', 'K', 'M']:
            col_name = f'star_type_{st}'
            if col_name not in df.columns:
                if 'star_type' in df.columns:
                    df[col_name] = (df['star_type'] == st).astype(int)
                else:
                    df[col_name] = 0
        
        # Prepare and predict
        available_features = [f for f in expected_features if f in df.columns]
        X = df[available_features].fillna(df[available_features].median())
        for feat in expected_features:
            if feat not in X.columns:
                X[feat] = 0.0
        X = X[expected_features]
        
        scaler, _ = load_scaler()
        X_scaled = scaler.transform(X) if scaler else X.values
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled) if hasattr(model, 'predict_proba') else None
        
        encoder, _ = load_label_encoder()
        if encoder and hasattr(encoder, 'inverse_transform'):
            predicted_classes = encoder.inverse_transform(predictions)
        else:
            predicted_classes = [str(p) for p in predictions]
        
        df['habitability_score'] = probabilities.max(axis=1) if probabilities is not None else predictions
        df['predicted_class'] = predicted_classes
        
        # Sort and select top N
        df_export = df.sort_values('habitability_score', ascending=False)
        top_n = int(request.args.get('top', 50))
        df_export = df_export.head(top_n)
        
        # Create PDF
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
        elements = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor('#2c3e50'))
        title = Paragraph("Top Habitable Exoplanets - ExoHabitatAI", title_style)
        elements.append(title)
        elements.append(Spacer(1, 0.2*inch))
        
        # Subtitle
        subtitle = Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal'])
        elements.append(subtitle)
        elements.append(Spacer(1, 0.3*inch))
        
        # Prepare table data
        table_data = [['Rank', 'Planet Name', 'Score', 'Class', 'Radius', 'Temp (K)', 'Star']]
        
        for idx, row in df_export.iterrows():
            rank = str(idx + 1)
            name = str(row.get('pl_name', 'Unknown'))[:25]
            score = f"{row.get('habitability_score', 0):.3f}"
            pred_class = str(row.get('predicted_class', 'N/A'))
            radius = f"{row.get('radius', 0):.2f}" if 'radius' in row else 'N/A'
            temp = f"{row.get('surface_temp', 0):.0f}" if 'surface_temp' in row else 'N/A'
            star = str(row.get('star_type', 'N/A'))
            
            table_data.append([rank, name, score, pred_class, radius, temp, star])
        
        # Create table
        table = Table(table_data, colWidths=[0.6*inch, 2.2*inch, 0.8*inch, 0.9*inch, 0.8*inch, 0.8*inch, 0.6*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        
        elements.append(table)
        
        # Build PDF
        doc.build(elements)
        buffer.seek(0)
        
        filename = f'top_habitable_exoplanets_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        
        return send_file(
            buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@api_bp.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint for monitoring and deployment platforms
    """
    # Test database connection
    db_status = db_manager.test_connection()
    
    return jsonify({
        'status': 'healthy',
        'version': '1.0.0',
        'message': 'ExoHabitatAI API is running',
        'database': db_status
    }), 200
