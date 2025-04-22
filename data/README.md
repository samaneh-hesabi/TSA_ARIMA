<div style="font-size:2.5em; font-weight:bold; text-align:center; margin-top:20px;">Data Directory</div>

# 1. Overview
This directory contains all datasets used in the Time Series Analysis project.

# 1.1 Required Files
- `bitcoin.csv`: Historical Bitcoin price data
  - Required columns: Date, Close
  - Date format: YYYY-MM-DD
  - Close: Numeric values representing closing prices

# 1.2 Data Source
You can obtain Bitcoin historical data from:
- Yahoo Finance
- CoinGecko
- CryptoCompare

# 1.3 Data Format
The CSV file should have the following structure:
```
Date,Close
2023-01-01,16500.00
2023-01-02,16750.00
...
```

# 1.4 Data Preparation
1. Download historical Bitcoin price data
2. Save it as `bitcoin.csv` in this directory
3. Ensure the file contains at least the Date and Close columns
4. Make sure the data is in chronological order

# 2. Directory Structure
- `raw/`: Original, unprocessed datasets
- `processed/`: Cleaned and transformed datasets
- `interim/`: Intermediate processing results
- `external/`: Third-party datasets

# 3. Data Management
## 3.1 File Naming Convention
Use the following format for dataset files:
```
[source]_[description]_[YYYYMMDD].[extension]
```
Example: `yahoo_finance_sp500_20240422.csv`

## 3.2 Data Formats
Supported formats:
- CSV (.csv)
- Excel (.xlsx, .xls)
- JSON (.json)
- Parquet (.parquet)

## 3.3 Data Documentation
Each dataset should include:
- Data dictionary
- Source information
- Collection date
- Update frequency
- Any preprocessing steps applied

# 4. Data Privacy
- Do not commit sensitive data
- Use .gitignore for large files
- Store credentials in environment variables
- Follow data protection regulations

# 5. Version Control
- Track data versions using timestamps
- Document changes between versions
- Maintain backup copies of important datasets 