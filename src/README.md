<div style="font-size:2.5em; font-weight:bold; text-align:center; margin-top:20px;">Time Series Analysis Source Code</div>

# 1. Overview
This directory contains the core Python source code for the Time Series Analysis project, specifically designed for financial time series analysis of stock market data.

# 2. Files
## 2.1 main.py
The entry point of the application that:
- Initializes the data processing pipeline with S&P 500 data
- Coordinates the analysis workflow
- Demonstrates the basic usage of the TimeSeriesProcessor and TimeSeriesAnalyzer classes

## 2.2 analysis.py
Contains the TimeSeriesAnalyzer class with comprehensive financial analysis capabilities:
- Basic dataset information and statistics
- Stationarity tests (ADF test)
- Normality tests (Shapiro-Wilk test)
- Autocorrelation analysis (ACF and PACF plots)
- Time series visualization (price, returns, volatility)
- Distribution analysis (histograms and Q-Q plots)
- Volume analysis and visualization

## 2.3 data_processor.py
Contains the TimeSeriesProcessor class for data handling:
- Data loading from Yahoo Finance
- Data cleaning and preprocessing
- Feature engineering:
  - Daily returns calculation
  - Log returns calculation
  - Moving averages (20-day and 50-day)
  - Volatility calculation
- Missing value handling

# 3. Usage
Basic usage example:
```python
from src.data_processor import TimeSeriesProcessor
from src.analysis import TimeSeriesAnalyzer

# Initialize processor and load data
processor = TimeSeriesProcessor(
    ticker='^GSPC',  # S&P 500 index
    start_date='2019-01-01',
    end_date='2024-01-01'
)

# Load and clean data
data = processor.load_data()
cleaned_data = processor.clean_data()

# Perform analysis
analyzer = TimeSeriesAnalyzer(processor)
analyzer.basic_analysis()
```

# 4. Dependencies
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- statsmodels
- yfinance
- logging

# 5. Development Guidelines
1. Follow PEP 8 style guidelines
2. Add comprehensive docstrings to all functions and classes
3. Include error handling and logging
4. Maintain consistent code structure
5. Update this documentation when adding new features
6. Write unit tests for new functionality 