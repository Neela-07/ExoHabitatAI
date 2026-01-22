"""
Module 7: Visualization & Dashboard
Build dashboard for visual insights:
- Feature importance plots
- Habitability score distribution
- Star-planet parameter correlations
Use matplotlib, seaborn, or plotly for interactive charts
Export reports (PDF/Excel) for top candidate exoplanets
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DATA_DIR, MODELS_DIR
from src.utils.database import DatabaseManager

class VisualizationDashboard:
    """
    Creates visualizations and dashboards for ExoHabitatAI
    """
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.output_dir = Path(__file__).parent.parent / "static" / "images"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_all_visualizations(self):
        """
        Generate all visualizations
        """
        print("Generating visualizations...")
        
        # Load data
        df = self.db_manager.load_data(source="processed")
        
        if df is None:
            print("Error: Could not load processed data.")
            return
        
        # Generate visualizations
        self.plot_habitability_distribution(df)
        self.plot_star_type_distribution(df)
        self.plot_feature_importance(df)
        self.plot_correlation_matrix(df)
        self.plot_score_distribution(df)
        self.plot_parameter_correlations(df)
        
        print("\nAll visualizations generated successfully!")
    
    def plot_habitability_distribution(self, df):
        """
        Plot habitability class distribution
        """
        print("\nGenerating habitability distribution plot...")
        
        if 'habitability_class' not in df.columns:
            print("  Warning: habitability_class not found in data")
            return
        
        # Count classes
        class_counts = df['habitability_class'].value_counts()
        
        # Matplotlib version
        plt.figure(figsize=(10, 6))
        colors_map = {'High': '#198754', 'Medium': '#ffc107', 'Low': '#fd7e14', 'Non-Habitable': '#dc3545'}
        class_counts.plot(kind='bar', color=[colors_map.get(c, '#6c757d') for c in class_counts.index])
        plt.title('Habitability Class Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Habitability Class', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'habitability_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plotly version (interactive)
        fig = px.pie(
            values=class_counts.values,
            names=class_counts.index,
            title='Habitability Class Distribution',
            color_discrete_map=colors_map
        )
        fig.write_html(str(self.output_dir / 'habitability_distribution.html'))
        
        print(f"  Saved to {self.output_dir / 'habitability_distribution.png'}")
    
    def plot_star_type_distribution(self, df):
        """
        Plot star type distribution
        """
        print("\nGenerating star type distribution plot...")
        
        if 'star_type' not in df.columns:
            print("  Warning: star_type not found in data")
            return
        
        star_counts = df['star_type'].value_counts().sort_index()
        
        plt.figure(figsize=(10, 6))
        star_counts.plot(kind='bar', color='#0d6efd')
        plt.title('Star Type Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Star Type', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'star_type_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved to {self.output_dir / 'star_type_distribution.png'}")
    
    def plot_feature_importance(self, df):
        """
        Plot feature importance (from model if available, otherwise correlation-based)
        """
        print("\nGenerating feature importance plot...")
        
        # Try to load model for actual feature importance
        try:
            import pickle
            model_path = MODELS_DIR / "best_model.pkl"
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                
                if hasattr(model, 'feature_importances_'):
                    feature_names = df.select_dtypes(include=[np.number]).columns.tolist()
                    # Filter out target columns
                    feature_names = [f for f in feature_names if 'habitability' not in f.lower() and 'predicted' not in f.lower()]
                    
                    # Match model features with dataset features
                    importances = model.feature_importances_[:len(feature_names)]
                    
                    # Create DataFrame
                    importance_df = pd.DataFrame({
                        'Feature': feature_names[:len(importances)],
                        'Importance': importances
                    }).sort_values('Importance', ascending=True)
                    
                    # Plot
                    plt.figure(figsize=(10, 8))
                    plt.barh(importance_df['Feature'], importance_df['Importance'], color='#198754')
                    plt.title('Feature Importance for Habitability Prediction', fontsize=16, fontweight='bold')
                    plt.xlabel('Importance', fontsize=12)
                    plt.ylabel('Feature', fontsize=12)
                    plt.tight_layout()
                    plt.savefig(self.output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"  Saved to {self.output_dir / 'feature_importance.png'}")
                    return
        except Exception as e:
            print(f"  Could not load model feature importance: {e}")
        
        # Fallback: Use correlation with habitability score
        if 'combined_habitability_score' in df.columns:
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numerical_cols = [c for c in numerical_cols if 'habitability' not in c.lower() and 'predicted' not in c.lower()]
            
            correlations = df[numerical_cols].corrwith(df['combined_habitability_score']).abs().sort_values(ascending=True)
            
            plt.figure(figsize=(10, 8))
            correlations.plot(kind='barh', color='#198754')
            plt.title('Feature Correlation with Habitability Score', fontsize=16, fontweight='bold')
            plt.xlabel('Absolute Correlation', fontsize=12)
            plt.ylabel('Feature', fontsize=12)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved to {self.output_dir / 'feature_importance.png'}")
    
    def plot_correlation_matrix(self, df):
        """
        Plot correlation matrix of key parameters
        """
        print("\nGenerating correlation matrix...")
        
        # Select numerical columns
        numerical_cols = ['radius', 'mass', 'density', 'surface_temp', 
                         'orbital_period', 'distance_from_star',
                         'star_luminosity', 'star_temp', 'metallicity']
        
        available_cols = [c for c in numerical_cols if c in df.columns]
        
        if len(available_cols) < 3:
            print("  Warning: Not enough numerical columns for correlation matrix")
            return
        
        corr_matrix = df[available_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Parameter Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved to {self.output_dir / 'correlation_matrix.png'}")
    
    def plot_score_distribution(self, df):
        """
        Plot habitability score distribution
        """
        print("\nGenerating score distribution plot...")
        
        score_col = None
        for col in ['combined_habitability_score', 'habitability_score_index', 'predicted_score']:
            if col in df.columns:
                score_col = col
                break
        
        if score_col is None:
            print("  Warning: No habitability score column found")
            return
        
        plt.figure(figsize=(10, 6))
        df[score_col].hist(bins=50, color='#0d6efd', edgecolor='black', alpha=0.7)
        plt.title('Habitability Score Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Habitability Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.axvline(df[score_col].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df[score_col].mean():.3f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'score_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved to {self.output_dir / 'score_distribution.png'}")
    
    def plot_parameter_correlations(self, df):
        """
        Plot star-planet parameter correlations
        """
        print("\nGenerating parameter correlation plots...")
        
        # Scatter plots: Star temperature vs Planet temperature
        if 'star_temp' in df.columns and 'surface_temp' in df.columns:
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.scatter(df['star_temp'], df['surface_temp'], alpha=0.5, color='#0d6efd')
            plt.xlabel('Star Temperature (K)', fontsize=12)
            plt.ylabel('Planet Surface Temperature (K)', fontsize=12)
            plt.title('Star vs Planet Temperature', fontsize=14, fontweight='bold')
            
            # Distance vs Temperature
            if 'distance_from_star' in df.columns:
                plt.subplot(1, 2, 2)
                plt.scatter(df['distance_from_star'], df['surface_temp'], alpha=0.5, color='#198754')
                plt.xlabel('Distance from Star (AU)', fontsize=12)
                plt.ylabel('Planet Surface Temperature (K)', fontsize=12)
                plt.title('Distance vs Temperature', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'parameter_correlations.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved to {self.output_dir / 'parameter_correlations.png'}")
    
    def export_pdf_report(self, df, top_n=20, filename='exoplanet_report.pdf'):
        """
        Export top candidate exoplanets to PDF
        """
        print(f"\nExporting PDF report with top {top_n} exoplanets...")
        
        # Sort by habitability score
        score_col = None
        for col in ['combined_habitability_score', 'habitability_score_index', 'predicted_score']:
            if col in df.columns:
                score_col = col
                break
        
        if score_col is None:
            print("  Error: No habitability score column found")
            return
        
        top_planets = df.nlargest(top_n, score_col)
        
        # Create PDF
        output_path = self.output_dir.parent.parent / filename
        doc = SimpleDocTemplate(str(output_path), pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title = Paragraph("Top Habitable Exoplanets Report", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 0.5*inch))
        
        # Table data
        table_data = [['Rank', 'Planet', 'Score', 'Class', 'Radius', 'Mass', 'Temp', 'Star Type']]
        
        for idx, (_, planet) in enumerate(top_planets.iterrows(), 1):
            row = [
                str(idx),
                str(planet.get('planet_name', f'Planet_{idx}')),
                f"{planet[score_col]:.3f}",
                str(planet.get('habitability_class', 'Unknown')),
                f"{planet.get('radius', 0):.2f}",
                f"{planet.get('mass', 0):.2f}",
                f"{int(planet.get('surface_temp', 0))}",
                str(planet.get('star_type', 'Unknown'))
            ]
            table_data.append(row)
        
        # Create table
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(table)
        
        # Build PDF
        doc.build(story)
        print(f"  Saved to {output_path}")
    
    def export_excel_report(self, df, top_n=20, filename='exoplanet_report.xlsx'):
        """
        Export top candidate exoplanets to Excel
        """
        print(f"\nExporting Excel report with top {top_n} exoplanets...")
        
        # Sort by habitability score
        score_col = None
        for col in ['combined_habitability_score', 'habitability_score_index', 'predicted_score']:
            if col in df.columns:
                score_col = col
                break
        
        if score_col is None:
            print("  Error: No habitability score column found")
            return
        
        top_planets = df.nlargest(top_n, score_col)
        
        # Select columns
        export_cols = ['planet_name', score_col, 'habitability_class', 
                      'radius', 'mass', 'density', 'surface_temp',
                      'orbital_period', 'distance_from_star',
                      'star_type', 'star_luminosity', 'star_temp', 'metallicity']
        
        available_cols = [c for c in export_cols if c in top_planets.columns]
        export_df = top_planets[available_cols].copy()
        export_df.insert(0, 'Rank', range(1, len(export_df) + 1))
        
        # Export to Excel
        output_path = self.output_dir.parent.parent / filename
        export_df.to_excel(output_path, index=False, sheet_name='Top Exoplanets')
        print(f"  Saved to {output_path}")

def main():
    """
    Main function for visualization dashboard
    """
    dashboard = VisualizationDashboard()
    
    # Generate all visualizations
    dashboard.generate_all_visualizations()
    
    # Export reports
    df = dashboard.db_manager.load_data(source="processed")
    if df is not None:
        dashboard.export_pdf_report(df, top_n=20)
        dashboard.export_excel_report(df, top_n=20)
    
    print("\nDashboard generation completed!")

if __name__ == "__main__":
    main()

