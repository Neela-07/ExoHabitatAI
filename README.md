# 🌍 ExoHabitatAI

**AI-Powered Exoplanet Habitability Prediction System**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Deploy](https://img.shields.io/badge/Deploy-Heroku%20%7C%20Render-purple.svg)](#-deployment)

A machine learning application that analyzes exoplanet data to predict habitability potential, featuring a Flask REST API, PostgreSQL database integration, and interactive web dashboard.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#features)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Machine Learning](#-machine-learning)
- [Database Configuration](#-database-configuration)
- [Deployment](#-deployment)
- [Environment Variables](#-environment-variables)
- [Testing](#-testing)
- [Documentation](#-documentation)
- [Contributing](#-contributing)
- [License](#-license)

---

## 📋 Overview

ExoHabitatAI processes planetary and stellar parameters through trained ML models to classify and rank exoplanets based on habitability scores. The system uses NASA Exoplanet Archive data and custom habitability scoring algorithms.

### Features

- 🤖 **Machine Learning**: Random Forest, XGBoost, Logistic Regression models
- 🌐 **REST API**: Flask backend with prediction and ranking endpoints
- 📊 **Dashboard**: Real-time visualizations with Plotly charts
- 📈 **Habitability Scoring**: Custom HSI (Habitability Score Index) algorithm
- 📄 **Export**: Download rankings as PDF or Excel reports
- 🗄️ **Database**: PostgreSQL support with CSV fallback
- ☁️ **Cloud Ready**: Deploy to Heroku or Render with one click

---

## 🏗️ Project Structure

```
ExoHabitatAI/
├── app.py                  # Flask application entry point
├── config.py               # Configuration settings
├── requirements.txt        # Python dependencies
├── Procfile                # Heroku deployment
├── render.yaml             # Render deployment
├── runtime.txt             # Python version specification
│
├── api/                    # REST API module
│   ├── __init__.py
│   └── routes.py           # API endpoints
│
├── src/                    # Source modules
│   ├── data_collection/    # Data fetching from NASA/Kaggle
│   │   ├── __init__.py
│   │   └── collector.py
│   ├── preprocessing/      # Data cleaning & feature engineering
│   │   ├── __init__.py
│   │   ├── data_cleaning.py
│   │   └── feature_engineering.py
│   ├── ml/                 # Machine learning models
│   │   ├── __init__.py
│   │   ├── data_preparation.py
│   │   └── train_models.py
│   └── utils/              # Utility functions
│       ├── __init__.py
│       └── database.py     # Database manager (PostgreSQL/CSV)
│
├── data/                   # Data storage
│   ├── exoplanets_full.csv # Complete dataset
│   ├── processed/          # Cleaned data
│   ├── raw/                # Original data
│   └── models/             # Trained ML models (.pkl)
│
├── templates/              # Jinja2 HTML templates
│   ├── index.html          # Home page
│   ├── dashboard.html      # Analytics dashboard
│   └── results.html        # Rankings page
│
├── static/                 # Static assets
│   ├── css/style.css
│   ├── js/main.js
│   └── js/dashboard.js
│
├── visualization/          # Chart generation
│   └── dashboard.py
│
├── scripts/                # Utility scripts
│   ├── run_pipeline.py     # Full ML pipeline
│   ├── predict_and_rank.py # Generate predictions
│   ├── analyze_dataset.py  # Data analysis
│   └── test_export.py      # Test exports
│
└── docs/                   # Documentation
    ├── PROJECT_OVERVIEW.md
    ├── QUICK_START.md
    ├── HABITABILITY_SCORE_INDEX.md
    └── EXPORT_GUIDE.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+ (recommended: 3.10+)
- pip or conda
- PostgreSQL (optional, for database mode)
- Git

### Local Installation

```bash
# Clone repository
git clone https://github.com/yourusername/ExoHabitatAI.git
cd ExoHabitatAI

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables (optional - for PostgreSQL)
# Windows:
set DATABASE_URL=postgresql://user:password@localhost:5432/exohabitat
# Linux/Mac:
export DATABASE_URL=postgresql://user:password@localhost:5432/exohabitat

# Run the application
python app.py
```

Open your browser: **http://localhost:5000**

---

## 💻 Usage

### Web Interface

| Page | URL | Description |
|------|-----|-------------|
| Home | `/` | Make predictions |
| Dashboard | `/dashboard` | View analytics |
| Rankings | `/results` | Top habitable exoplanets |

### Rankings Page Features

- **Show Top Filter**: Select 10, 25, 50, or 100 planets
- **Export PDF**: Download PDF report
- **Export Excel**: Download spreadsheet
- **Sortable Table**: View planet details

---

## 📡 API Reference

### Base URL

- **Local**: `http://localhost:5000/api`
- **Production**: `https://your-app.herokuapp.com/api` or `https://your-app.onrender.com/api`

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/predict` | POST | Predict habitability |
| `/api/planets` | GET | Get all exoplanet data |
| `/api/rankings?top=N` | GET | Get top N ranked planets |
| `/api/statistics` | GET | Dataset statistics |
| `/api/export/pdf?top=N` | GET | Export rankings as PDF |
| `/api/export/excel?top=N` | GET | Export rankings as Excel |

### Example: Health Check

```bash
curl http://localhost:5000/api/health
```

Response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "database": "connected"
}
```

### Example: Predict Habitability

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "radius": 1.2,
    "mass": 2.5,
    "density": 5.5,
    "surface_temp": 288,
    "orbital_period": 365,
    "distance_from_star": 1.0,
    "star_type": "G",
    "star_temp": 5778,
    "metallicity": 0.0
  }'
```

Response:
```json
{
  "status": "success",
  "habitability_score": 0.85,
  "habitability_class": "High",
  "confidence": 0.92,
  "model_used": "xgboost"
}
```

### Example: Get Rankings

```bash
curl http://localhost:5000/api/rankings?top=10
```

### Example: Export Reports

```bash
# Download PDF
curl -O http://localhost:5000/api/export/pdf?top=50

# Download Excel
curl -O http://localhost:5000/api/export/excel?top=100
```

---

## 🧠 Machine Learning

### Models

| Model | Purpose |
|-------|---------|
| Random Forest | Primary classifier |
| XGBoost | High-accuracy predictions |
| Logistic Regression | Baseline comparison |

### Features Used

**Planetary:**
- Radius, Mass, Density
- Surface Temperature
- Orbital Period
- Distance from Star

**Stellar:**
- Star Type (O, B, A, F, G, K, M)
- Star Temperature
- Luminosity
- Metallicity

**Engineered:**
- Habitability Score Index (HSI)
- Stellar Compatibility Index (SCI)
- Radius/Distance Ratio
- Mass/Radius Ratio

### Performance

- Accuracy: 85-92%
- Precision: 84-90%
- ROC-AUC: 0.88-0.94

---

## ⚙️ Configuration

### Local Configuration

Edit `config.py` to customize settings:

```python
# Database type: "postgresql" or "csv"
DATABASE_CONFIG = {
    "type": "csv",  # Change to "postgresql" for database mode
    "postgresql": {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", 5432)),
        "database": os.getenv("DB_NAME", "exohabitat"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", "")
    }
}

# Flask settings
FLASK_CONFIG = {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": True  # Set to False in production
}
```

---

## 🗄️ Database Configuration

### Option 1: CSV Mode (Default)

No setup required. Data is stored in CSV files in the `data/` directory.

### Option 2: PostgreSQL Mode

1. **Install PostgreSQL** on your system or use a cloud provider

2. **Create Database**:
```sql
CREATE DATABASE exohabitat;
```

3. **Set Environment Variables**:
```bash
# Windows
set DATABASE_URL=postgresql://user:password@localhost:5432/exohabitat

# Linux/Mac
export DATABASE_URL=postgresql://user:password@localhost:5432/exohabitat
```

4. **Update config.py**:
```python
DATABASE_CONFIG = {
    "type": "postgresql",
    ...
}
```

5. **Initialize Database** (run once):
```bash
python scripts/run_pipeline.py --init-db
```

---

## ☁️ Deployment

### Deploy to Heroku

1. **Install Heroku CLI**: https://devcenter.heroku.com/articles/heroku-cli

2. **Login and Create App**:
```bash
heroku login
heroku create exohabitat-ai
```

3. **Add PostgreSQL Database**:
```bash
heroku addons:create heroku-postgresql:essential-0
```

4. **Set Environment Variables**:
```bash
heroku config:set FLASK_ENV=production
heroku config:set SECRET_KEY=your-secret-key-here
```

5. **Deploy**:
```bash
git push heroku main
```

6. **Initialize Database**:
```bash
heroku run python scripts/run_pipeline.py --init-db
```

7. **Open App**:
```bash
heroku open
```

### Deploy to Render

1. **Create Account**: https://render.com

2. **Connect GitHub Repository**

3. **Create New Web Service**:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`

4. **Add PostgreSQL Database**:
   - Create a new PostgreSQL service in Render
   - Copy the Internal Database URL

5. **Set Environment Variables**:
   - `DATABASE_URL`: Your PostgreSQL connection string
   - `FLASK_ENV`: `production`
   - `SECRET_KEY`: Your secret key
   - `PYTHON_VERSION`: `3.10.0`

6. **Deploy**: Render will auto-deploy on git push

### Deploy with Docker

```bash
# Build image
docker build -t exohabitat-ai .

# Run container
docker run -p 5000:5000 \
  -e DATABASE_URL=postgresql://user:pass@host:5432/db \
  exohabitat-ai
```

---

## 🔐 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection URL | None (uses CSV) |
| `DB_HOST` | Database host | `localhost` |
| `DB_PORT` | Database port | `5432` |
| `DB_NAME` | Database name | `exohabitat` |
| `DB_USER` | Database user | `postgres` |
| `DB_PASSWORD` | Database password | Empty |
| `FLASK_ENV` | Environment (development/production) | `development` |
| `SECRET_KEY` | Flask secret key | Auto-generated |
| `PORT` | Server port | `5000` |

---

## 📦 Dependencies

### Core Dependencies
```
flask>=2.0.0
flask-cors>=3.0.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
```

### Database
```
psycopg2-binary>=2.9.0
sqlalchemy>=1.4.0
```

### Visualization
```
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0
```

### Export
```
openpyxl>=3.0.0
xlsxwriter>=3.0.0
reportlab>=3.6.0
```

### Production
```
gunicorn>=20.1.0
```

Install all: `pip install -r requirements.txt`

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Test API endpoints
python scripts/test_export.py

# Test health endpoint
curl http://localhost:5000/api/health

# Test prediction
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"radius": 1.0, "mass": 1.0, "density": 5.5}'
```

---

## 📚 Documentation

- [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md) - Full details
- [docs/QUICK_START.md](docs/QUICK_START.md) - Setup guide
- [docs/HABITABILITY_SCORE_INDEX.md](docs/HABITABILITY_SCORE_INDEX.md) - HSI algorithm
- [docs/EXPORT_GUIDE.md](docs/EXPORT_GUIDE.md) - Export features

---

## 📊 Data

- **Source**: NASA Exoplanet Archive, Kaggle
- **Records**: 219,000+ exoplanets
- **Features**: 15+ planetary/stellar parameters

---

## 🤝 Contributing

1. Fork repository
2. Create branch: `git checkout -b feature/name`
3. Commit: `git commit -m 'Add feature'`
4. Push: `git push origin feature/name`
5. Open Pull Request

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
- [Kaggle Exoplanet Datasets](https://www.kaggle.com/)
- [scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [Flask](https://flask.palletsprojects.com/)
- [Plotly](https://plotly.com/)

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/ExoHabitatAI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/ExoHabitatAI/discussions)

---

**Made with ❤️ for exoplanet research**

*Last updated: January 2026*

