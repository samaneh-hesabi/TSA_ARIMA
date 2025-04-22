<div style="font-size:2.5em; font-weight:bold; text-align:center; margin-top:20px;">Time Series Analysis of Bitcoin Price Data</div>

# 1. Project Overview
This is a Time Series Analysis project focused on Bitcoin price prediction using ARIMA (AutoRegressive Integrated Moving Average) modeling. The project provides a systematic approach to analyzing and forecasting Bitcoin prices using historical data.

# 2. Detailed Project Explanation

## 2.1 Data Source and Structure
- The project uses historical Bitcoin price data stored in `btc-usd-max.csv`
- The data contains daily price records with timestamps
- The dataset includes 4,377 data points, providing a substantial historical record for analysis
- Data is sourced from CoinGecko's historical price API

## 2.2 Analysis Methodology
The project follows a systematic approach to time series analysis:

### 2.2.1 Data Preprocessing
- Data cleaning and formatting
- Conversion of timestamps to proper datetime format
- Ensuring price data is in numeric format (handling comma-separated values)
- Sorting data chronologically
- Setting timestamps as the index for time series analysis

### 2.2.2 Statistical Analysis
- **Stationarity Testing**: Using the Augmented Dickey-Fuller test to determine if the time series is stationary
  - This is crucial because ARIMA models require stationary data
  - If non-stationary, differencing is needed to make it stationary
  - Results are displayed with ADF statistic and p-value

### 2.2.3 Model Development
- **ARIMA Model**: Uses a (1,0,1) configuration where:
  - 1: Autoregressive term (p)
  - 0: Differencing term (d)
  - 1: Moving average term (q)
- The model is fitted to the historical price data
- Comprehensive model summary is provided including:
  - AIC/BIC values
  - Coefficient estimates
  - Standard errors
  - P-values for coefficients

### 2.2.4 Visualization Components
The project generates several important visualizations:
- **ACF (Autocorrelation Function) Plot**: Shows correlation between observations at different time lags (40 lags)
- **PACF (Partial Autocorrelation Function) Plot**: Shows the correlation between observations at different lags, controlling for intermediate observations (40 lags)
- **Residual Plots**: 
  - Time series plot of residuals
  - ACF plot of residuals to check for remaining patterns
- **Forecast Visualization**: Shows the predicted Bitcoin prices for the next 10 days with confidence intervals

## 2.3 Practical Applications
This analysis can be valuable for:
- Understanding Bitcoin price patterns
- Making short-term price predictions
- Identifying trends and seasonality in cryptocurrency markets
- Building a foundation for more complex forecasting models
- Educational purposes in time series analysis

## 2.4 Limitations and Considerations
- The model uses a simple ARIMA configuration (1,0,1) which might need tuning for better accuracy
- Cryptocurrency markets are highly volatile, so predictions should be used with caution
- The model doesn't account for external factors that might affect Bitcoin prices
- The 10-day forecast horizon is relatively short-term
- The model assumes stationarity in the time series
- No hyperparameter tuning is performed for the ARIMA model

# 3. Technical Requirements
The project requires the following Python packages:
- pandas: For data manipulation and analysis
- numpy: For numerical computations
- matplotlib: For creating visualizations
- statsmodels: For statistical modeling and time series analysis
- Python 3.6 or higher

# 4. Usage Instructions
1. Install the required Python packages:
   ```bash
   pip install pandas numpy matplotlib statsmodels
   ```
2. Ensure the `btc-usd-max.csv` file is in the same directory as `process.py`
3. Run the `process.py` script:
   ```bash
   python process.py
   ```
4. Review the generated visualizations and statistical outputs
5. Interpret the results and forecasts

# 5. Files Description

## 5.1 Data Files
- `btc-usd-max.csv`: Historical Bitcoin price data in USD format
  - Contains daily price data with timestamps
  - Used as input for time series analysis
  - Data format: CSV with columns 'snapped_at' (timestamp) and 'price' (USD)

## 5.2 Script Files
- `process.py`: Main analysis script that performs:
  - Data loading and preprocessing
  - Stationarity testing using Augmented Dickey-Fuller test
  - ACF and PACF analysis (40 lags)
  - ARIMA model fitting and forecasting
  - Residual analysis
  - 10-day price forecasting
  - Visualization generation

# 6. Output
The script generates several outputs:
1. Console Output:
   - Sample data preview
   - Augmented Dickey-Fuller test results
   - ARIMA model summary
   - 10-day forecast values
2. Visualizations:
   - ACF and PACF plots
   - Residual time series plot
   - Residual ACF plot
   - 10-day price forecast visualization with confidence intervals 