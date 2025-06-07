import pandas as pd                  # For data manipulation and analysis
import yfinance as yf                # For downloading financial data
import matplotlib.pyplot as plt      # For creating visualizations
import numpy as np                   # For numerical operations
from datetime import datetime, timedelta        # For handling date and time
from statsmodels.tsa.stattools import adfuller    # For stationarity testing(Augmented Dickey-Fuller (ADF) Test)
from statsmodels.tsa.arima.model import ARIMA     # For ARIMA modeling
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # For autocorrelation plots
from sklearn.metrics import mean_squared_error, mean_absolute_error  # For model evaluation
import itertools                     # For creating parameter combinations
import warnings                      # For handling warnings
import os                          # For file operations
warnings.filterwarnings('ignore')    # Suppress warnings to keep output clean
##===================================
##===================================
# STEP 1: Load and prepare data
print("Step 1: Loading and preparing data...")
ticker = '^GSPC'  # S&P 500 index symbol
start_date = '2019-01-01'
end_date = datetime.now().strftime('%Y-%m-%d')

# Function to create sample data if needed
def create_sample_data():
    print("Creating sample stock data for testing...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1000)  # ~3 years of data
    
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    
    # Create a sample price series with realistic properties
    np.random.seed(42)  # For reproducibility
    prices = [1000]  # Starting price
    for _ in range(1, len(date_range)):
        # Random walk with drift
        change = np.random.normal(0.0005, 0.01) * prices[-1]
        prices.append(prices[-1] + change)
    
    # Create DataFrame
    df = pd.DataFrame({'Close': prices}, index=date_range)
    return df

# Data loading options:
# Option 1: Try to load from CSV file first
# Option 2: If no CSV, try to download from Yahoo Finance
# Option 3: If Yahoo Finance fails, use sample data

csv_path = 'sp500_data.csv'
use_sample_data = False  # Flag to determine if we're using sample data

try:
    # First try to load from CSV if it exists
    if os.path.exists(csv_path):
        print(f"Loading data from {csv_path}")
        data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    else:
        # If no CSV exists, try to download data
        print(f"Loading {ticker} data from {start_date} to {end_date}")
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        
        # Save to CSV for future use
        data.to_csv(csv_path)
    
    # Keep only the closing prices
    data = data[['Close']]
    
except Exception as e:
    print(f"Error loading data: {e}")
    print("Using generated sample data instead...")
    data = create_sample_data()
    use_sample_data = True

# Fill missing values with previous day's value
data = data.fillna(method='ffill')
print("Data prepared successfully")

##===================================
##===================================

# STEP 2: Check stationarity
print("\nStep 2: Checking stationarity...")
# Augmented Dickey-Fuller test to check if data is stationary
result = adfuller(data['Close'].dropna())
print(result)
print('ADF Statistic:', result[0])  # Test statistic
print('p-value:', result[1])        # p-value

# Interpret results
if result[1] <= 0.05:
    print("Data is stationary (reject H0)")    # If p-value <= 0.05, data is stationary
else:
    print("Data is not stationary (fail to reject H0)")    # If p-value > 0.05, data is not stationary
    print("Differencing is needed to make the data stationary")    # If data is not stationary, differencing is needed to make it stationary
    
# Make the data stationary by differencing
diff_data = data['Close'].diff().dropna()

# Check stationarity of differenced data
diff_result = adfuller(diff_data.dropna())
print('\nDifferenced data:')
print('ADF Statistic:', diff_result[0])
print('p-value:', diff_result[1])

# Determine the appropriate differencing level (d)
if diff_result[1] <= 0.05:
    print("Differenced data is stationary (good for ARIMA)")
    d_value = 1  # First differencing is sufficient
else:
    # Try second differencing if first is not enough
    print("Differenced data is still not stationary, trying second differencing")
    diff2_data = diff_data.diff().dropna()   # Second difference
    diff2_result = adfuller(diff2_data.dropna())
    print('\nSecond differenced data:')
    print('ADF Statistic:', diff2_result[0])
    print('p-value:', diff2_result[1])
    
    # Determine if second differencing helps
    if diff2_result[1] <= 0.05:
        print("Second differenced data is stationary")
        d_value = 2     # Need second differencing
    else:
        print("Using first differencing anyway as it's most common")
        d_value = 1     # Default to first differencing

##===================================
##===================================
# STEP 3: Visualize the data
print("\nStep 3: Visualizing data...")
# Plot original vs differenced data
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Original data
ax1.plot(data['Close'])
ax1.set_title('Original Time Series')
ax1.grid(True)

# Differenced data
ax2.plot(diff_data)
ax2.set_title('Differenced Time Series')
ax2.grid(True)

plt.tight_layout()
plt.show()

##===================================
##===================================

# STEP 4: Analyze ACF and PACF to identify ARIMA parameters
print("\nStep 4: Analyzing ACF and PACF to identify ARIMA parameters...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot ACF
plot_acf(diff_data, ax=ax1, lags=20)
ax1.set_title('Autocorrelation Function (ACF)')
ax1.grid(True)

# Plot PACF
plot_pacf(diff_data, ax=ax2, lags=20)
ax2.set_title('Partial Autocorrelation Function (PACF)')
ax2.grid(True)

plt.tight_layout()
plt.show()

print("\nHow to interpret ACF/PACF plots for ARIMA parameter selection:")
print("- p (AR order): Look at PACF plot - how many significant lags before it cuts off")
print("- d (differencing): Already determined by stationarity tests")
print("- q (MA order): Look at ACF plot - how many significant lags before it cuts off")
print(f"Based on stationarity tests, d = {d_value}")

##===================================
##===================================
# STEP 5: Grid search for best ARIMA parameters
print("\nStep 5: Finding the best ARIMA parameters...")

# Split data into train and test sets
train_size = int(len(data) * 0.8)
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]
print(f"Training data: {len(train_data)} days, Test data: {len(test_data)} days")

# Define the parameter grid to search
p_values = range(0, 3)  # AR parameter options: 0, 1, 2
d_values = [d_value]    # Use the differencing value determined earlier
q_values = range(0, 3)  # MA parameter options: 0, 1, 2

# Initialize variables to store results
best_aic = float('inf')     # Start with infinity (we want to minimize AIC)
best_bic = float('inf')
best_params = None
best_model = None

results_list = []

# Grid search - try all combinations of parameters
for p, d, q in itertools.product(p_values, d_values, q_values):
    try:
        print(f"Trying ARIMA({p},{d},{q})...")
        model = ARIMA(train_data['Close'], order=(p, d, q))
        results = model.fit()
        
        # Store results
        aic = results.aic
        bic = results.bic
        
        print(f"  AIC: {aic:.2f}, BIC: {bic:.2f}")
        
        # Save all results
        results_list.append((p, d, q, aic, bic))
        
        # Check if this is the best model
        if aic < best_aic:
            best_aic = aic
            best_bic = bic
            best_params = (p, d, q)
            best_model = results
            
    except Exception as e:
        print(f"  Error with ARIMA({p},{d},{q}): {str(e)}")
        continue

# Create a dataframe with all results
models_df = pd.DataFrame(results_list, 
                         columns=['p', 'd', 'q', 'AIC', 'BIC'])
models_df = models_df.sort_values('AIC')
                         
print("\nModel comparison (sorted by AIC):")
print(models_df.head(10))

print(f"\nBest model parameters: ARIMA{best_params}")
print(f"Best AIC: {best_aic:.2f}")
print(f"Best BIC: {best_bic:.2f}")

##===================================
##===================================
# STEP 6: Evaluate the best model
print("\nStep 6: Evaluating the best model...")

# Get model summary
print("\nBest model summary:")
print(best_model.summary())

# Get predictions for test period
start_idx = len(train_data)
end_idx = len(train_data) + len(test_data) - 1
predictions = best_model.predict(start=start_idx, end=end_idx)

# Calculate model performance metrics
test_actual = test_data['Close'].values    # Actual values
test_predictions = predictions.values      # Predicted values

# Calculate metrics
mae = mean_absolute_error(test_actual, test_predictions)
mse = mean_squared_error(test_actual, test_predictions)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((test_actual - test_predictions) / test_actual)) * 100

print("\nPerformance metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Plot test vs predictions
plt.figure(figsize=(12, 6))
plt.plot(test_data.index, test_actual, label='Actual')
plt.plot(test_data.index, test_predictions, color='red', label='Predicted')
plt.title(f'ARIMA{best_params} Model Performance on Test Data')
plt.xlabel('Date')
plt.ylabel('Stock Price ($)')
plt.legend()
plt.grid(True)
plt.show()

##===================================
##===================================
# STEP 7: Residual analysis
print("\nStep 7: Residual analysis...")
# Check if residuals are white noise (important for ARIMA validity)
residuals = best_model.resid[1:]

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

# Plot residuals
ax1.plot(residuals)
ax1.set_title('Residuals')
ax1.set_xlabel('Time')
ax1.set_ylabel('Residual Value')
ax1.grid(True)

# Plot residual histogram
ax2.hist(residuals, bins=30)
ax2.set_title('Residual Histogram')
ax2.set_xlabel('Residual Value')
ax2.set_ylabel('Frequency')
ax2.grid(True)

# Plot residual ACF
plot_acf(residuals, ax=ax3, lags=20)
ax3.set_title('Residual ACF')
ax3.grid(True)

plt.tight_layout()
plt.show()

# Check if residuals are normally distributed
from scipy import stats
stat, p_value = stats.normaltest(residuals)
print("\nNormality test of residuals:")
print(f"p-value: {p_value:.4f}")
if p_value < 0.05:
    print("Residuals are not normally distributed - model may not be optimal")
else:
    print("Residuals are normally distributed - good model fit")

##===================================
##===================================
# STEP 8: Forecast future values
print("\nStep 8: Forecasting future values...")
# Retrain model on full dataset with best parameters
p, d, q = best_params
print(f"Retraining ARIMA({p},{d},{q}) on full dataset...")
full_model = ARIMA(data['Close'], order=(p, d, q))
full_results = full_model.fit()

# Forecast future values
steps = 30  # Number of days to forecast
forecast = full_results.forecast(steps=steps)
forecast_ci = full_results.get_forecast(steps=steps).conf_int()

# Create forecast index (dates)
last_date = data.index[-1]
forecast_index = pd.date_range(start=last_date, periods=steps+1)[1:]

# Plot the forecast with confidence intervals
plt.figure(figsize=(12, 6))
plt.plot(data.index[-90:], data['Close'][-90:], label='Historical Data')
plt.plot(forecast_index, forecast, color='red', label='Forecast')
plt.fill_between(forecast_index, 
                 forecast_ci.iloc[:, 0], 
                 forecast_ci.iloc[:, 1], 
                 color='pink', alpha=0.3, label='95% Confidence Interval')
plt.title(f'ARIMA{best_params} Forecast - Next {steps} Days')
plt.xlabel('Date')
plt.ylabel('Stock Price ($)')
plt.legend()
plt.grid(True)
plt.show()

print("\nAnalysis complete!") 