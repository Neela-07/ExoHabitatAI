"""
Module 4: AI Model for Habitability Prediction
Train models:
- Random Forest Classifier for binary habitability prediction
- XGBoost Classifier for multi-class habitability levels
- Logistic Regression for comparison
Evaluate models using:
- Accuracy, Precision, Recall, F1-score, ROC-AUC
Rank exoplanets based on predicted habitability scores
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import xgboost as xgb
import sys
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import MODEL_CONFIG, MODELS_DIR, RANDOM_STATE
from src.ml.data_preparation import MLDataPreparation

class ModelTrainer:
    """
    Trains and evaluates machine learning models for habitability prediction
    """
    
    def __init__(self):
        self.data_prep = MLDataPreparation()
        self.models = {}
        self.results = {}
        
    def train_models(self):
        """
        Main training pipeline
        """
        print("Starting model training...")
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.data_prep.prepare_data()
        
        if X_train is None:
            print("Error: Data preparation failed.")
            return
        
        # Train Random Forest
        print("\n" + "="*60)
        print("Training Random Forest Classifier...")
        print("="*60)
        rf_model = self.train_random_forest(X_train, y_train, X_test, y_test)
        
        # Train XGBoost
        print("\n" + "="*60)
        print("Training XGBoost Classifier...")
        print("="*60)
        xgb_model = self.train_xgboost(X_train, y_train, X_test, y_test)
        
        # Train Logistic Regression
        print("\n" + "="*60)
        print("Training Logistic Regression...")
        print("="*60)
        lr_model = self.train_logistic_regression(X_train, y_train, X_test, y_test)
        
        # Compare models
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        self.compare_models()
        
        # Save best model (based on F1-score)
        self.save_best_model()
        
        print("\nModel training completed!")
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """
        Train Random Forest Classifier
        """
        config = MODEL_CONFIG["random_forest"]
        
        model = RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            random_state=config["random_state"],
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Evaluate
        metrics = self.evaluate_model(y_test, y_pred, y_pred_proba, "Random Forest")
        self.results["Random Forest"] = metrics
        
        # Save model
        model_path = Path(config["path"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"\nModel saved to {model_path}")
        
        self.models["Random Forest"] = model
        return model
    
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """
        Train XGBoost Classifier
        """
        config = MODEL_CONFIG["xgboost"]
        
        # Determine if multiclass
        n_classes = len(np.unique(y_train))
        
        # Check if we have only one class
        if n_classes == 1:
            print("\n⚠️  WARNING: Only one class present in training data.")
            print("Skipping XGBoost training - cannot train classifier with single class.")
            # Create a dummy model that always predicts the single class
            from sklearn.dummy import DummyClassifier
            model = DummyClassifier(strategy='constant', constant=y_train[0])
        else:
            objective = 'multi:softprob' if n_classes > 2 else 'binary:logistic'
            model = xgb.XGBClassifier(
                n_estimators=config["n_estimators"],
                max_depth=config["max_depth"],
                learning_rate=config["learning_rate"],
                random_state=config["random_state"],
                objective=objective,
                n_jobs=-1
            )
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Evaluate
        metrics = self.evaluate_model(y_test, y_pred, y_pred_proba, "XGBoost")
        self.results["XGBoost"] = metrics
        
        # Save model
        model_path = Path(config["path"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"\nModel saved to {model_path}")
        
        self.models["XGBoost"] = model
        return model
    
    def train_logistic_regression(self, X_train, y_train, X_test, y_test):
        """
        Train Logistic Regression
        """
        n_classes = len(np.unique(y_train))
        
        # Handle single class case
        if n_classes == 1:
            print("\n⚠️  WARNING: Only one class present in training data.")
            print("Skipping Logistic Regression training - cannot train classifier with single class.")
            from sklearn.dummy import DummyClassifier
            model = DummyClassifier(strategy='constant', constant=y_train[0])
        else:
            model = LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Evaluate
        metrics = self.evaluate_model(y_test, y_pred, y_pred_proba, "Logistic Regression")
        self.results["Logistic Regression"] = metrics
        
        # Save model
        model_path = MODELS_DIR / "logistic_regression_model.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"\nModel saved to {model_path}")
        
        self.models["Logistic Regression"] = model
        return model
    
    def evaluate_model(self, y_true, y_pred, y_pred_proba, model_name):
        """
        Evaluate model performance
        """
        metrics = {}
        
        # Classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # ROC-AUC (for binary or multiclass)
        if y_pred_proba is not None:
            try:
                n_classes = len(np.unique(y_true))
                if n_classes == 2:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
            except:
                metrics['roc_auc'] = 0.0
        
        # Print results
        print(f"\n{model_name} Performance:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        if 'roc_auc' in metrics:
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))
        
        return metrics
    
    def compare_models(self):
        """
        Compare all trained models
        """
        comparison_df = pd.DataFrame(self.results).T
        print("\n" + comparison_df.to_string())
        
        # Find best model by F1-score
        best_model = comparison_df['f1_score'].idxmax()
        print(f"\nBest Model (by F1-Score): {best_model}")
        print(f"  F1-Score: {comparison_df.loc[best_model, 'f1_score']:.4f}")
        
        return comparison_df
    
    def save_best_model(self):
        """
        Save the best performing model as default
        """
        if not self.results:
            return
        
        comparison_df = pd.DataFrame(self.results).T
        best_model_name = comparison_df['f1_score'].idxmax()
        
        best_model_path = MODELS_DIR / "best_model.pkl"
        best_model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy best model
        best_model = self.models[best_model_name]
        with open(best_model_path, 'wb') as f:
            pickle.dump(best_model, f)
        
        # Save metadata
        metadata = {
            'model_name': best_model_name,
            'metrics': self.results[best_model_name]
        }
        metadata_path = MODELS_DIR / "model_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"\nBest model ({best_model_name}) saved to {best_model_path}")
    
    def rank_exoplanets(self, df, model_name="XGBoost"):
        """
        Rank exoplanets based on predicted habitability scores
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found. Loading from file...")
            model_path = MODEL_CONFIG.get(model_name.lower().replace(" ", "_"), {}).get("path")
            if model_path and Path(model_path).exists():
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            else:
                print("Model file not found.")
                return None
        else:
            model = self.models[model_name]
        
        # Prepare features
        feature_columns = self.data_prep.get_feature_names()
        X = df[feature_columns].fillna(df[feature_columns].median())
        
        # Load scaler
        scaler_path = MODELS_DIR / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X
        
        # Predict
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled) if hasattr(model, 'predict_proba') else None
        
        # Rank by probability of highest class
        if probabilities is not None:
            max_probability = probabilities.max(axis=1)
            df['predicted_habitability_score'] = max_probability
            df['predicted_habitability_class'] = predictions
        
        # Sort by score
        ranked_df = df.sort_values('predicted_habitability_score', ascending=False)
        
        return ranked_df

def main():
    """
    Main function for model training
    """
    trainer = ModelTrainer()
    trainer.train_models()

if __name__ == "__main__":
    main()

