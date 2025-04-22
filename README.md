<div style="font-size:2.5em; font-weight:bold; text-align:center; margin-top:20px;">Time Series Analysis Project</div>

# 1. Project Overview
This project provides a comprehensive toolkit for financial time series analysis, focusing on stock market data analysis and visualization. The project is particularly focused on analyzing the S&P 500 index, but can be extended to analyze any financial instrument available through Yahoo Finance.

# 2. Project Structure
```
time-series-analysis/
├── src/                    # Main source code directory for S&P 500 analysis
│   ├── main.py            # Main entry point for S&P 500 analysis
│   ├── analysis.py        # Time series analysis functions
│   ├── data_processor.py  # Data processing utilities
│   └── README.md          # Source code documentation
├── scr2/                   # Bitcoin price analysis directory
│   ├── process.py         # Main script for Bitcoin price analysis
│   ├── btc-usd-max.csv    # Bitcoin historical price data
│   └── README.md          # Bitcoin analysis documentation
├── notebooks/             # Jupyter notebooks for analysis
├── data/                  # Data storage directory
├── environment.yml        # Conda environment configuration
├── requirements.txt       # Python package dependencies
└── .gitignore            # Git ignore configuration
```

# 3. Features and Capabilities

## 3.1 S&P 500 Analysis (src directory)
- Automated data fetching from Yahoo Finance
- Data cleaning and preprocessing
- Feature engineering:
  - Daily returns calculation
  - Log returns calculation
  - Moving averages (20-day and 50-day)
  - Volatility calculation (20-day rolling)
- Basic statistical analysis
- Stationarity testing (Augmented Dickey-Fuller test)
- Normality testing (Shapiro-Wilk test)
- Autocorrelation analysis (ACF and PACF plots)
- Volume analysis
- Distribution analysis

## 3.2 Bitcoin Analysis (scr2 directory)
- Historical Bitcoin price data analysis
- ARIMA (1,0,1) modeling for price prediction
- Stationarity testing using Augmented Dickey-Fuller test
- ACF and PACF analysis
- 10-day price forecasting
- Residual analysis
- Comprehensive visualization of results

## 3.3 Visualization
- Time series plots with moving averages
- Returns distribution plots
- Q-Q plots for normality assessment
- Volume analysis plots
- Autocorrelation function plots

# 4. Dataset Information

## 4.1 S&P 500 Data

| Feature | Description | Data Type | Example |
|---------|-------------|-----------|---------|
| Open | Opening price | float | 4000.25 |
| High | Highest price of the day | float | 4020.50 |
| Low | Lowest price of the day | float | 3990.75 |
| Close | Closing price | float | 4010.00 |
| Volume | Trading volume | int | 2500000 |
| Returns | Daily percentage returns | float | 0.0025 |
| Log_Returns | Natural log of returns | float | 0.0025 |
| MA20 | 20-day moving average | float | 3980.50 |
| MA50 | 50-day moving average | float | 3950.25 |
| Volatility | 20-day rolling volatility | float | 0.015 |

### 4.1.1 S&P 500 Analysis Explanation

The S&P 500 analysis focuses on traditional financial market analysis with the following key components:

1. **Price Analysis**
   - Daily price movements (Open, High, Low, Close)
   - Volume analysis to understand trading activity
   - Moving averages (20-day and 50-day) for trend identification
   - Volatility measurement using 20-day rolling standard deviation

2. **Returns Analysis**
   - Daily percentage returns calculation
   - Log returns for better statistical properties
   - Distribution analysis of returns
   - Normality testing using Shapiro-Wilk test

3. **Technical Analysis**
   - Moving average crossovers for trend signals
   - Volume-price relationship analysis
   - Volatility clustering analysis
   - Support and resistance level identification

4. **Statistical Analysis**
   - Stationarity testing using Augmented Dickey-Fuller test
   - Autocorrelation analysis (ACF and PACF plots)
   - Distribution fitting and analysis
   - Outlier detection and analysis

## 4.2 Bitcoin Data

| Feature | Description | Data Type | Example |
|---------|-------------|-----------|---------|
| Timestamp | Date and time of the price record | datetime | 2023-01-01 00:00:00 |
| Price | Bitcoin price in USD | float | 42000.50 |
| Returns | Daily percentage returns | float | 0.015 |
| Log_Returns | Natural log of returns | float | 0.0149 |
| ARIMA_Prediction | Predicted price from ARIMA model | float | 42500.75 |
| Residual | Difference between actual and predicted price | float | -500.25 |
| ACF | Autocorrelation at different lags | float | 0.85 |
| PACF | Partial autocorrelation at different lags | float | 0.45 |

### 4.2.1 Bitcoin Analysis Explanation

The Bitcoin analysis focuses on time series forecasting using ARIMA modeling with the following key components:

1. **Data Preprocessing**
   - Timestamp handling and chronological ordering
   - Price data normalization
   - Returns calculation (both simple and logarithmic)
   - Data cleaning and outlier handling

2. **ARIMA Modeling**
   - Model configuration: ARIMA(1,0,1)
     - 1 autoregressive term
     - 0 differencing terms
     - 1 moving average term
   - Model fitting and parameter estimation
   - Residual analysis for model validation
   - 10-day price forecasting

3. **Statistical Analysis**
   - Stationarity testing using Augmented Dickey-Fuller test
   - Autocorrelation analysis (ACF plots)
   - Partial autocorrelation analysis (PACF plots)
   - Residual diagnostics and model validation

4. **Forecasting and Validation**
   - 10-day price predictions
   - Confidence intervals for forecasts
   - Model accuracy metrics
   - Residual analysis for model improvement

### 4.2.2 Key Differences Between Analyses

1. **Methodology**
   - S&P 500: Traditional financial analysis with technical indicators
   - Bitcoin: Advanced time series modeling with ARIMA

2. **Focus**
   - S&P 500: Market behavior and technical analysis
   - Bitcoin: Price prediction and forecasting

3. **Data Characteristics**
   - S&P 500: More stable, traditional market data
   - Bitcoin: Higher volatility, cryptocurrency-specific patterns

4. **Analysis Tools**
   - S&P 500: Moving averages, volume analysis, volatility measures
   - Bitcoin: ARIMA modeling, autocorrelation analysis, forecasting

# 5. Installation

## 5.1 Prerequisites
- Python 3.8 or higher
- Conda package manager

## 5.2 Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/time-series-analysis.git
   cd time-series-analysis
   ```

2. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate time-series-env
   ```

3. Install additional requirements:
   ```bash
   pip install -r requirements.txt
   ```

## 5.3 Dependencies
The project requires the following Python packages:
- pandas (>=2.0.0)
- numpy (>=1.24.0)
- matplotlib (>=3.7.0)
- seaborn (>=0.12.0)
- scikit-learn (>=1.2.0)
- statsmodels (>=0.14.0)
- jupyter (>=1.0.0)
- python-dotenv (>=1.0.0)

# 6. Usage

## 6.1 Basic Usage
Run the main script to analyze S&P 500 data:
```bash
python src/main.py
```

## 6.2 Custom Analysis
To analyze a different stock or time period, modify the parameters in `main.py`:
```python
processor = TimeSeriesProcessor(
    ticker='AAPL',  # Change to desired ticker
    start_date='2020-01-01',  # Change start date
    end_date='2023-12-31'  # Change end date
)
```

# 7. Results and Output

The analysis generates several types of outputs:

1. **Statistical Summary**
   - Basic statistics (mean, std, min, max, etc.)
   - Stationarity test results
   - Normality test results

2. **Visualizations**
   - Price chart with moving averages
   - Returns distribution
   - Q-Q plot for normality
   - Volume analysis
   - Autocorrelation plots

3. **Analysis Reports**
   - Trading day statistics
   - Missing value counts
   - Volume analysis

## 7.1 Result Interpretation

### 7.1.1 Statistical Analysis Results

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| Mean Return | Average daily return | Positive values indicate overall upward trend |
| Standard Deviation | Volatility measure | Higher values indicate more risk |
| Skewness | Distribution asymmetry | Negative skew suggests more frequent large negative returns |
| Kurtosis | Distribution "tailedness" | Values > 3 indicate fat tails (more extreme events) |

### 7.1.2 Stationarity Tests
- **Augmented Dickey-Fuller (ADF) Test**
  - p-value > 0.05: Series is non-stationary
  - p-value ≤ 0.05: Series is stationary
  - Expected: Price series typically non-stationary, returns series stationary

### 7.1.3 Normality Tests
- **Shapiro-Wilk Test**
  - p-value > 0.05: Returns follow normal distribution
  - p-value ≤ 0.05: Returns deviate from normal distribution
  - Expected: Financial returns typically show non-normal distribution

### 7.1.4 Moving Averages Analysis
- **20-day vs 50-day MA Crossover**
  - 20-day MA > 50-day MA: Bullish signal
  - 20-day MA < 50-day MA: Bearish signal
  - Crossovers indicate potential trend changes

### 7.1.5 Volume Analysis
- **Volume Trends**
  - Increasing volume with price: Confirms trend
  - Decreasing volume with price: Potential trend reversal
  - Unusual volume spikes: May indicate significant events

### 7.1.6 Autocorrelation Analysis
- **ACF/PACF Plots**
  - Significant lags: Indicate serial correlation
  - No significant lags: Random walk behavior
  - Expected: Financial returns typically show little autocorrelation

### 7.1.7 Volatility Analysis
- **20-day Rolling Volatility**
  - Higher values: Increased market uncertainty
  - Lower values: Market stability
  - Clustering: Volatility tends to cluster in time

## 7.2 Example Output Interpretation

```
=== Dataset Information ===
Time Period: 2019-01-01 to 2023-12-31
Total Trading Days: 1258
Missing Values: 0

=== Basic Statistics ===
Returns:
mean      0.0004
std       0.0123
min      -0.1276
max       0.0958
skew     -0.45
kurtosis  8.23

=== Stationarity Tests ===
ADF Test for Returns - p-value: 0.0001

=== Normality Tests ===
Shapiro-Wilk Test - p-value: 0.0000
```

**Interpretation:**
1. The dataset covers approximately 5 years of daily trading data
2. Returns show:
   - Slight positive mean return (0.04% daily)
   - Moderate volatility (1.23% daily)
   - Negative skewness (more frequent large negative returns)
   - High kurtosis (fat tails, more extreme events)
3. Returns are stationary (p-value < 0.05)
4. Returns are not normally distributed (p-value < 0.05)

# 8. Contributing

1. Fork the repository
2. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "feat: add your feature description"
   ```
4. Push to the branch:
   ```bash
   git push origin feature/your-feature-name
   ```
5. Open a pull request

# 9. License
This project is licensed under the MIT License - see the LICENSE file for details.

# 10. Acknowledgments
- Yahoo Finance API for providing financial data
- Python scientific computing community for the excellent libraries used in this project


