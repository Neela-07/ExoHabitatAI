"""
Test script for PDF and Excel export functionality
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

def test_excel_export():
    """Test Excel export functionality"""
    print("Testing Excel export...")
    try:
        # Create sample data
        data = {
            'Rank': range(1, 11),
            'Planet Name': [f'Planet_{i}' for i in range(1, 11)],
            'Score': [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45],
            'Class': ['High'] * 3 + ['Medium'] * 4 + ['Low'] * 3,
            'Radius': [1.2, 1.5, 1.1, 0.9, 1.3, 1.4, 1.0, 0.8, 1.6, 1.1],
            'Mass': [2.5, 3.0, 2.2, 1.8, 2.7, 2.9, 2.0, 1.5, 3.2, 2.3],
            'Temperature': [288, 295, 280, 310, 290, 285, 300, 320, 275, 305]
        }
        
        df = pd.DataFrame(data)
        
        # Export to Excel
        output_file = Path(__file__).parent.parent / 'test_export.xlsx'
        df.to_excel(output_file, index=False, sheet_name='Test Data')
        
        print(f"✓ Excel export successful: {output_file}")
        print(f"  File size: {output_file.stat().st_size} bytes")
        return True
    except Exception as e:
        print(f"✗ Excel export failed: {e}")
        return False

def test_pdf_export():
    """Test PDF export functionality"""
    print("\nTesting PDF export...")
    try:
        # Create sample PDF
        output_file = Path(__file__).parent.parent / 'test_export.pdf'
        doc = SimpleDocTemplate(str(output_file), pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title = Paragraph("ExoHabitatAI - Test PDF Export", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 0.3 * inch))
        
        # Subtitle
        subtitle = Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal'])
        story.append(subtitle)
        story.append(Spacer(1, 0.3 * inch))
        
        # Table data
        table_data = [
            ['Rank', 'Planet', 'Score', 'Class', 'Radius', 'Mass', 'Temp'],
            ['1', 'Planet_1', '0.900', 'High', '1.20', '2.50', '288'],
            ['2', 'Planet_2', '0.850', 'High', '1.50', '3.00', '295'],
            ['3', 'Planet_3', '0.800', 'High', '1.10', '2.20', '280'],
            ['4', 'Planet_4', '0.750', 'Medium', '0.90', '1.80', '310'],
            ['5', 'Planet_5', '0.700', 'Medium', '1.30', '2.70', '290'],
        ]
        
        # Create table
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(table)
        
        # Build PDF
        doc.build(story)
        
        print(f"✓ PDF export successful: {output_file}")
        print(f"  File size: {output_file.stat().st_size} bytes")
        return True
    except Exception as e:
        print(f"✗ PDF export failed: {e}")
        return False

def main():
    """Run all export tests"""
    print("=" * 60)
    print("ExoHabitatAI - Export Functionality Test")
    print("=" * 60)
    
    excel_success = test_excel_export()
    pdf_success = test_pdf_export()
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"  Excel Export: {'✓ PASS' if excel_success else '✗ FAIL'}")
    print(f"  PDF Export:   {'✓ PASS' if pdf_success else '✗ FAIL'}")
    print("=" * 60)
    
    if excel_success and pdf_success:
        print("\n✓ All export tests passed!")
        print("\nYou can now use the export buttons in the web interface:")
        print("  1. Start the Flask app: python app.py")
        print("  2. Navigate to: http://localhost:5000/results")
        print("  3. Click 'Export PDF' or 'Export Excel' buttons")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())
