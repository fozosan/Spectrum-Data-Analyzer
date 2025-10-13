# Raman Analyzer

A desktop application for post-peak Raman spectroscopy analysis based on CSV exports from peak-fitting tools.

## Installation

```bash
pip install -r requirements.txt
```

## Running the application

```bash
python app.py
```

## CSV schema

The application expects CSV files containing peak measurements with at least a `file` column or infers the file name from the CSV path. Typical columns include `peak_id`, `peak_index`, `center`, `height`, `area`, `fwhm`, and `area_pct`.

## Roadmap

- Additional trendline models (quadratic, power).
- Expanded plotting types (line, violin) and residual exports.
- Session persistence and enhanced data export options.
