# ExoHabitatAI - Project Overview

## 📋 Complete Project Structure

```
ExoHabitatAI/
├── app.py                          # Main Flask application
├── config.py                       # Configuration settings
├── requirements.txt                # Python dependencies
├── README.md                       # Comprehensive documentation
├── PROJECT_OVERVIEW.md            # This file - project overview
├── .gitignore                     # Git ignore rules
│
├── data/                          # Data directory
│   ├── raw/                       # Raw datasets (gitignored)
│   ├── processed/                 # Processed datasets (gitignored)
│   └── models/                    # Trained ML models (gitignored)
│       ├── random_forest_model.pkl
│       ├── xgboost_model.pkl
│       ├── logistic_regression_model.pkl
│       ├── best_model.pkl
│       ├── scaler.pkl
│       ├── label_encoder.pkl
│       └── model_metadata.pkl
│
├── src/                           # Source code modules
│   ├── __init__.py
│   ├── data_collection/          # Module 1: Data Collection
│   │   ├── __init__.py
│   │   └── collector.py          # Collects exoplanet data from NASA/Kaggle
│   │
│   ├── preprocessing/             # Module 2: Data Cleaning & Feature Engineering
│   │   ├── __init__.py
│   │   ├── data_cleaning.py      # Handles missing values, outliers
│   │   └── feature_engineering.py # Creates HSI, SCI, encodes features
│   │
│   ├── ml/                        # Module 3 & 4: ML Preparation & Models
│   │   ├── __init__.py
│   │   ├── data_preparation.py   # Train/test split, feature selection
│   │   └── train_models.py       # Trains RF, XGBoost, Logistic Regression
│   │
│   └── utils/                     # Utility functions
│       ├── __init__.py
│       └── database.py           # Database manager (PostgreSQL/CSV)
│
├── api/                           # Module 5: Flask Backend API
│   ├── __init__.py
│   └── routes.py                 # REST API endpoints
│       ├── POST /api/predict     # Predict habitability
│       ├── GET /api/planets      # Retrieve exoplanet data
│       ├── GET /api/rankings     # Get habitability rankings
│       ├── GET /api/statistics   # Get dataset statistics
│       └── GET /api/health       # Health check
│
├── templates/                     # Module 6: Frontend UI (HTML)
│   ├── index.html               # Home page with prediction form
│   ├── dashboard.html           # Analytics dashboard
│   └── results.html             # Habitability rankings table
│
├── static/                        # Static assets
│   ├── css/
│   │   └── style.css            # Custom CSS styling
│   ├── js/
│   │   ├── main.js              # Main JavaScript for index.html
│   │   ├── dashboard.js         # Dashboard JavaScript
│   │   └── results.js           # Results/rankings JavaScript
│   └── images/                   # Generated visualization images
│       ├── habitability_distribution.png
│       ├── star_type_distribution.png
│       ├── feature_importance.png
│       ├── correlation_matrix.png
│       └── score_distribution.png
│
├── visualization/                 # Module 7: Visualization & Dashboard
│   └── dashboard.py             # Generates charts, PDF/Excel reports
│
└── tests/                        # Unit tests (to be added)
```

## 🎯 Module Implementation Summary

### Milestone 1: Week 1-2 ✅

#### Module 1: Data Collection and Management
- **File**: `src/data_collection/collector.py`
- **Features**:
  - Collects data from NASA Exoplanet Archive API
  - Creates sample datasets if external sources unavailable
  - Validates schema and data completeness
  - Stores in PostgreSQL or CSV format
  - Handles data validation and quality checks

#### Module 2: Data Cleaning and Feature Engineering
- **Files**: 
  - `preprocessing/data_cleaning.py`
  - `preprocessing/feature_engineering.py`
- **Features**:
  - Handles missing values (median/mode imputation)
  - Outlier detection and capping using IQR
  - Feature engineering:
    - **Habitability Score Index (HSI)**: Based on temperature, radius, density, distance
    - **Stellar Compatibility Index (SCI)**: Based on star type, temperature, metallicity
  - One-hot encoding for categorical features
  - Data type validation and normalization
  - Creates habitability class labels (High/Medium/Low/Non-Habitable)

### Milestone 2: Week 3-4 ✅

#### Module 3: Machine Learning Dataset Preparation
- **File**: `src/ml/data_preparation.py`
- **Features**:
  - Train/test split (80:20) with stratification
  - Feature selection based on correlation with habitability
  - Data pipelines with scaling (StandardScaler)
  - Label encoding for categorical targets
  - Saves scaler and encoder for prediction

#### Module 4: AI Model for Habitability Prediction
- **File**: `src/ml/train_models.py`
- **Models Implemented**:
  1. **Random Forest Classifier**
     - n_estimators=100, max_depth=10
  2. **XGBoost Classifier**
     - n_estimators=100, max_depth=6, learning_rate=0.1
  3. **Logistic Regression**
     - Multi-class support (multinomial/ovr)
- **Evaluation Metrics**:
  - Accuracy, Precision, Recall, F1-score
  - ROC-AUC (binary and multiclass)
  - Classification reports
  - Model comparison and best model selection
- **Features**:
  - Ranks exoplanets by predicted habitability scores
  - Saves best model automatically
  - Model metadata storage

### Milestone 3: Week 5-6 ✅

#### Module 5: Flask Backend API
- **File**: `api/routes.py`, `app.py`
- **Endpoints**:
  - `POST /api/predict`: Accept exoplanet parameters, return prediction
  - `GET /api/planets`: Retrieve exoplanet data (paginated)
  - `GET /api/rankings`: Get top habitable exoplanets (ranked)
  - `GET /api/statistics`: Get dataset statistics
  - `GET /api/health`: Health check endpoint
- **Features**:
  - JSON request/response format
  - Error handling and validation
  - Model loading and prediction
  - Database integration (CSV/PostgreSQL)
  - CORS enabled for frontend access

#### Module 6: Frontend UI Development
- **Files**:
  - `templates/index.html`: Main prediction interface
  - `templates/dashboard.html`: Analytics dashboard
  - `templates/results.html`: Rankings table
  - `static/css/style.css`: Custom styling
  - `static/js/main.js`: Form handling and API calls
  - `static/js/dashboard.js`: Dashboard visualizations
  - `static/js/results.js`: Rankings table management
- **Features**:
  - Responsive Bootstrap 5 design
  - Clean, modern UI with animations
  - Interactive prediction form
  - Real-time API integration
  - DataTables for rankings
  - Plotly charts for visualizations

### Milestone 4: Week 7-8 ✅

#### Module 7: Visualization & Dashboard
- **File**: `visualization/dashboard.py`
- **Visualizations**:
  1. **Habitability Distribution**: Pie/bar chart of class distribution
  2. **Star Type Distribution**: Bar chart of star types
  3. **Feature Importance**: Horizontal bar chart (from model or correlation)
  4. **Correlation Matrix**: Heatmap of parameter correlations
  5. **Score Distribution**: Histogram of habitability scores
  6. **Parameter Correlations**: Scatter plots (star vs planet temp, distance vs temp)
- **Export Features**:
  - **PDF Export**: Top N exoplanets with formatted table (ReportLab)
  - **Excel Export**: Top N exoplanets with all parameters (openpyxl)
- **Output Formats**:
  - Static images (PNG, 300 DPI)
  - Interactive HTML charts (Plotly)
  - PDF reports
  - Excel spreadsheets

## 🚀 Quick Start Guide

### 1. Installation
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Pipeline Execution Order

```bash
# Step 1: Collect data
python src/data_collection/collector.py

# Step 2: Clean data
python preprocessing/data_cleaning.py

# Step 3: Engineer features
python preprocessing/feature_engineering.py

# Step 4: Prepare ML dataset
python src/ml/data_preparation.py

# Step 5: Train models
python src/ml/train_models.py

# Step 6: Generate visualizations
python visualization/dashboard.py
```

### 3. Run Flask Application
```bash
python app.py
```

Then open browser: `http://localhost:5000`

## 📊 Output Screenshots & Features

### Web Interface Features:
- ✅ **Prediction Form**: Input planetary and stellar parameters
- ✅ **Real-time Predictions**: Get habitability class and confidence score
- ✅ **Dashboard**: Interactive charts and statistics
- ✅ **Rankings Table**: Sortable, searchable table of top exoplanets
- ✅ **Export Options**: PDF and Excel export buttons

### Visualizations Generated:
- ✅ Feature importance plots
- ✅ Habitability score distribution
- ✅ Star-planet parameter correlations
- ✅ Correlation matrices
- ✅ Class distributions

### Reports Generated:
- ✅ PDF reports with top candidate exoplanets
- ✅ Excel spreadsheets with detailed data

## 🔧 Configuration

Edit `config.py` to configure:
- Database type (PostgreSQL or CSV)
- Model hyperparameters
- Flask server settings
- File paths
- Feature columns
- Habitability thresholds

## 📝 Key Features

1. **Comprehensive Data Pipeline**: From raw data to predictions
2. **Multiple ML Models**: Random Forest, XGBoost, Logistic Regression
3. **Robust Evaluation**: Multiple metrics for model comparison
4. **Production-Ready API**: RESTful endpoints with error handling
5. **Modern UI**: Responsive Bootstrap design with interactive elements
6. **Rich Visualizations**: Multiple chart types with Plotly
7. **Export Capabilities**: PDF and Excel report generation

## 🎓 Educational Value

This project demonstrates:
- End-to-end ML pipeline
- Data preprocessing and feature engineering
- Model training and evaluation
- Flask API development
- Frontend/backend integration
- Data visualization
- Report generation

## 📦 Dependencies

All dependencies are listed in `requirements.txt`:
- Flask & Flask-CORS
- pandas, numpy
- scikit-learn
- xgboost
- matplotlib, seaborn, plotly
- reportlab (PDF), openpyxl (Excel)
- psycopg2-binary (PostgreSQL support)

## 🎯 Next Steps

To extend the project:
1. Add unit tests (`tests/` directory)
2. Add more visualization types
3. Implement real-time model retraining
4. Add user authentication
5. Deploy to cloud (AWS, Heroku, etc.)
6. Add more data sources
7. Implement ensemble models
8. Add interactive 3D visualizations

---

**Project Status**: ✅ Complete - All modules implemented according to specifications

**Last Updated**: 2024

