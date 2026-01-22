# Export Functionality Guide

## Overview

ExoHabitatAI supports exporting habitability rankings to both **PDF** and **Excel** formats. This allows researchers and users to save, share, and further analyze the prediction results.

---

## Features

### PDF Export
- Professional formatted document
- Includes:
  - Report title and generation timestamp
  - Top N habitable exoplanets in a clean table
  - Planet name, habitability score, class, radius, mass, temperature, and star type
  - Color-coded layout for easy reading

### Excel Export
- Spreadsheet format (`.xlsx`)
- Includes:
  - All planetary data in tabular format
  - Suitable for further analysis in Excel, Python, R, etc.
  - Filterable and sortable columns
  - Professional formatting

---

## How to Use

### From the Web Interface

#### 1. Rankings Page (`/results`)

1. Navigate to: `http://localhost:5000/results`
2. Select how many top planets to display using the dropdown (10, 25, 50, 100)
3. Click one of the export buttons in the table header:
   - **Export PDF**: Downloads a PDF report
   - **Export Excel**: Downloads an Excel spreadsheet
4. The file will be downloaded automatically with a timestamp in the filename

#### 2. Dashboard Page (`/dashboard`)

1. Navigate to: `http://localhost:5000/dashboard`
2. Click the export buttons in the "Quick Links" section:
   - **Export PDF**: Downloads top 50 planets by default
   - **Export Excel**: Downloads top 100 planets by default

---

## API Endpoints

### Export to PDF
```bash
GET /api/export/pdf?top=<number>
```

**Parameters:**
- `top` (optional): Number of top planets to export (default: 50)

**Example:**
```bash
curl -O http://localhost:5000/api/export/pdf?top=20
```

**Response:**
- Content-Type: `application/pdf`
- Filename: `top_habitable_exoplanets_YYYYMMDD_HHMMSS.pdf`

---

### Export to Excel
```bash
GET /api/export/excel?top=<number>
```

**Parameters:**
- `top` (optional): Number of top planets to export (default: 100)

**Example:**
```bash
curl -O http://localhost:5000/api/export/excel?top=50
```

**Response:**
- Content-Type: `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet`
- Filename: `top_habitable_exoplanets_YYYYMMDD_HHMMSS.xlsx`

---

## Export Data Structure

### Columns Included

| Column | Description | Example |
|--------|-------------|---------|
| Rank | Planet ranking by habitability | 1, 2, 3... |
| Planet Name | Name of the exoplanet | Kepler-442b |
| Habitability Score | Predicted score (0-1) | 0.892 |
| Predicted Class | Classification | High, Medium, Low |
| Radius | Planet radius (Earth radii) | 1.34 |
| Mass | Planet mass (Earth masses) | 2.36 |
| Density | Planet density (g/cm³) | 5.51 |
| Temperature | Surface temperature (K) | 288 |
| Orbital Period | Days to orbit star | 112.3 |
| Distance from Star | Distance in AU | 1.0 |
| Star Temperature | Star temperature (K) | 5778 |
| Star Type | Spectral type | G, K, M, etc. |

---

## File Naming Convention

Exported files follow this naming pattern:
```
top_habitable_exoplanets_YYYYMMDD_HHMMSS.[pdf|xlsx]
```

**Example:**
- `top_habitable_exoplanets_20260117_143025.pdf`
- `top_habitable_exoplanets_20260117_143025.xlsx`

This ensures unique filenames and prevents overwriting previous exports.

---

## Testing Export Functionality

Run the test script to verify exports work correctly:

```bash
cd F:\ExoHabitatAI\ExoHabitatAI
python scripts\test_export.py
```

Expected output:
```
============================================================
ExoHabitatAI - Export Functionality Test
============================================================
Testing Excel export...
✓ Excel export successful
  File size: 5905 bytes

Testing PDF export...
✓ PDF export successful
  File size: 2294 bytes

Test Summary:
  Excel Export: ✓ PASS
  PDF Export:   ✓ PASS
============================================================
```

---

## Dependencies

The following Python packages are required for export functionality:

### PDF Export
- `reportlab` - PDF generation library

### Excel Export
- `openpyxl` - Excel file creation and manipulation
- `xlsxwriter` - Alternative Excel writer (optional)

### Installation

All dependencies are included in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install reportlab openpyxl xlsxwriter
```

---

## Troubleshooting

### Issue: Export buttons not working

**Solution:**
1. Check that Flask server is running
2. Open browser console (F12) to check for JavaScript errors
3. Verify the API endpoints are accessible:
   ```bash
   curl http://localhost:5000/api/export/pdf?top=5
   ```

### Issue: PDF export fails

**Solution:**
1. Ensure `reportlab` is installed:
   ```bash
   pip install reportlab
   ```
2. Check that data is available in the database
3. Review Flask console for error messages

### Issue: Excel export fails

**Solution:**
1. Ensure `openpyxl` is installed:
   ```bash
   pip install openpyxl
   ```
2. Verify sufficient disk space for the export
3. Check Flask logs for detailed error messages

### Issue: No data in exported files

**Solution:**
1. Ensure ML models are trained and predictions exist
2. Run the data pipeline:
   ```bash
   python scripts/run_pipeline.py
   ```
3. Check that processed data exists in `data/processed/`

---

## Advanced Usage

### Python Script Export

You can also use the export functionality programmatically:

```python
import requests

# Export PDF
response = requests.get('http://localhost:5000/api/export/pdf?top=20')
with open('my_export.pdf', 'wb') as f:
    f.write(response.content)

# Export Excel
response = requests.get('http://localhost:5000/api/export/excel?top=50')
with open('my_export.xlsx', 'wb') as f:
    f.write(response.content)
```

### Batch Export

Create exports for different ranges:

```bash
# Top 10, 50, 100, 500 planets
curl -O http://localhost:5000/api/export/pdf?top=10
curl -O http://localhost:5000/api/export/pdf?top=50
curl -O http://localhost:5000/api/export/pdf?top=100
curl -O http://localhost:5000/api/export/pdf?top=500
```

---

## Best Practices

1. **Use appropriate limits**: Export only what you need to keep file sizes manageable
2. **Save regularly**: Export results after each model training session
3. **Version control**: Include timestamps in filenames for tracking
4. **Backup exports**: Keep copies of important export files
5. **Combine formats**: Use Excel for analysis, PDF for presentations

---

## Future Enhancements

Planned improvements for export functionality:

- [ ] Custom column selection
- [ ] Multiple output formats (CSV, JSON)
- [ ] Charts and visualizations in PDF
- [ ] Batch export scheduling
- [ ] Email delivery of reports
- [ ] Cloud storage integration
- [ ] Interactive PDF with hyperlinks

---

## Support

For issues or questions about export functionality:

1. Check this documentation first
2. Run the test script: `python scripts/test_export.py`
3. Review Flask logs for error messages
4. Open an issue on GitHub with:
   - Error message
   - Browser console output
   - Flask server logs

---

**Last Updated:** January 17, 2026
