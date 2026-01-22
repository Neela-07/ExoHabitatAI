# Quick Start Guide - ExoHabitatAI

## Fastest Way to Run the Project

### Option 1: Use the Automated Pipeline Script (Recommended)

```bash
# 1. Activate virtual environment
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 2. Run the complete pipeline
python run_pipeline.py
```

This will automatically run all steps in sequence!

### Option 2: Run Steps Manually

Follow the detailed guide in `RUN_PROJECT.md` for step-by-step instructions.

## Minimal Steps to See Output

If you want to quickly see the web interface working:

```bash
# 1. Activate virtual environment
venv\Scripts\activate

# 2. Install dependencies (if not done)
pip install -r requirements.txt

# 3. Run data collection (uses existing data if available)
python src/data_collection/collector.py

# 4. Run preprocessing
python preprocessing/data_cleaning.py
python preprocessing/feature_engineering.py

# 5. Train models (this is required for predictions)
python src/ml/train_models.py

# 6. Start Flask server
python app.py
```

Then open browser: `http://localhost:5000`

## Quick Test Commands

**Test if Flask is working:**
```bash
python app.py
# In another terminal/command prompt:
curl http://localhost:5000/api/health
```

**Test data collection:**
```bash
python src/data_collection/collector.py
```

**Test prediction (after training models):**
- Open browser: http://localhost:5000
- Fill in the form and click "Predict Habitability"

## Expected Runtime

- Data Collection: ~30 seconds - 2 minutes (depends on data source)
- Data Cleaning: ~10-30 seconds
- Feature Engineering: ~10-30 seconds  
- ML Preparation: ~5-10 seconds
- Model Training: ~1-5 minutes (depends on dataset size)
- Flask Server: Starts immediately

**Total time for first run: ~3-8 minutes**

## Troubleshooting

**Problem:** Import errors
**Solution:** Make sure you're in the project root directory and virtual environment is activated

**Problem:** Port 5000 in use
**Solution:** Change port in `config.py` or stop other application using port 5000

**Problem:** Models not found
**Solution:** Make sure you ran `python src/ml/train_models.py` first

**Problem:** Data files not found
**Solution:** Run `python src/data_collection/collector.py` first

## What You'll See

### During Pipeline Execution:
- Progress messages for each step
- Data statistics
- Model performance metrics
- Success/error messages

### In Web Browser:
- **Home Page**: Prediction form with planet/star parameters
- **Dashboard**: Charts and statistics
- **Rankings**: Top habitable exoplanets table

### API Response:
- JSON responses with predictions
- Status codes and error messages
- Data in structured format

---

**For detailed instructions, see `RUN_PROJECT.md`**

