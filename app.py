"""
Main Flask Application for ExoHabitatAI
Supports deployment on Heroku, Render, and local development
"""
if _name_ == "_main_":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
    
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import sys
import os
from pathlib import Path

# Import API routes
from api.routes import api_bp

# Import configuration
sys.path.append(str(Path(__file__).parent))
from config import FLASK_CONFIG, IS_PRODUCTION

app = Flask(__name__)
app.secret_key = FLASK_CONFIG.get('secret_key', 'dev-secret-key')
CORS(app)  # Enable CORS for all routes

# Register API blueprint
app.register_blueprint(api_bp)

@app.route('/')
def index():
    """
    Home page
    """
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """
    Dashboard page
    """
    return render_template('dashboard.html')

@app.route('/results')
def results():
    """
    Results page
    """
    return render_template('results.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    """
    Serve static files
    """
    return send_from_directory('static', filename)

if __name__ == '__main__':
    print("="*60)
    print("ExoHabitatAI - Exoplanet Habitability Prediction System")
    print("="*60)
    print(f"Environment: {'Production' if IS_PRODUCTION else 'Development'}")
    print(f"Starting Flask server on {FLASK_CONFIG['host']}:{FLASK_CONFIG['port']}")
    print(f"Debug mode: {FLASK_CONFIG['debug']}")
    print("="*60)
    
    app.run(
        host=FLASK_CONFIG['host'],
        port=FLASK_CONFIG['port'],
        debug=FLASK_CONFIG['debug']
    )
