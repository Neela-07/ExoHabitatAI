"""
Quick Start Script - Runs the complete ExoHabitatAI pipeline
This script executes all steps in sequence
"""
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and display progress"""
    print("\n" + "="*60)
    print(f"STEP: {description}")
    print("="*60)
    result = os.system(command)
    if result != 0:
        print(f"\n⚠️  WARNING: {description} returned exit code {result}")
        print("You may need to run this step manually to see detailed errors.")
    return result == 0

def main():
    """Main execution function"""
    print("="*60)
    print("ExoHabitatAI - Complete Pipeline Runner")
    print("="*60)
    print("\nThis script will run all pipeline steps in sequence.")
    print("Press Ctrl+C to cancel at any time.\n")
    
    # Check if running from correct directory
    if not Path("config.py").exists():
        print("❌ ERROR: config.py not found!")
        print("Please run this script from the project root directory.")
        return
    
    # Step 1: Data Collection
    if not run_command(
        "python src/data_collection/collector.py",
        "Module 1: Data Collection and Management"
    ):
        print("\n⚠️  Data collection had issues. Continuing anyway...")
    
    # Step 2: Data Cleaning
    if not run_command(
        "python preprocessing/data_cleaning.py",
        "Module 2 (Part 1): Data Cleaning"
    ):
        print("\n⚠️  Data cleaning had issues. Continuing anyway...")
    
    # Step 3: Feature Engineering
    if not run_command(
        "python preprocessing/feature_engineering.py",
        "Module 2 (Part 2): Feature Engineering"
    ):
        print("\n⚠️  Feature engineering had issues. Continuing anyway...")
    
    # Step 4: ML Data Preparation
    if not run_command(
        "python src/ml/data_preparation.py",
        "Module 3: ML Dataset Preparation"
    ):
        print("\n⚠️  ML preparation had issues. Continuing anyway...")
    
    # Step 5: Train Models
    if not run_command(
        "python src/ml/train_models.py",
        "Module 4: Train ML Models"
    ):
        print("\n⚠️  Model training had issues. Please check errors above.")
        return
    
    # Step 6: Generate Visualizations (Optional)
    print("\n" + "="*60)
    response = input("Generate visualizations? (y/n, default=y): ").strip().lower()
    if response != 'n':
        run_command(
            "python visualization/dashboard.py",
            "Module 7: Generate Visualizations"
        )
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETED!")
    print("="*60)
    print("\nNext steps:")
    print("1. Start Flask server: python app.py")
    print("2. Open browser: http://localhost:5000")
    print("3. Test predictions using the web interface")
    print("\n" + "="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Pipeline execution cancelled by user.")
        sys.exit(1)

