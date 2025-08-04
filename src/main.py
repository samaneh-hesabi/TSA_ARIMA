"""
Simple Stock Price Analysis and Prediction using ARIMA
This script performs time series analysis on stock price data using the ARIMA model.
It handles data preparation, model training, evaluation, and future price prediction.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA  # For time series modeling
from statsmodels.tsa.stattools import adfuller  # For stationarity test
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import warnings

# Suppress warning messages for cleaner output
warnings.filterwarnings('ignore')

def load_data(file_path):
    """Load stock price data from CSV file."""
    # Read CSV file with dates as index
    data = pd.read_csv(file_path, index_col=0, parse_dates=True)
    # Extract closing prices and handle missing values
    return data['Close'].fillna(method='ffill')

def make_stationary(data):
    """Make time series stationary through differencing.
    
    Stationarity is important for ARIMA modeling. A stationary time series
    has constant statistical properties over time (mean, variance).
    """
    # Step 1: Test original data for stationarity
    # adfuller returns (statistic, p-value, lags, observations)
    # p-value <= 0.05 means data is stationary
    if adfuller(data)[1] <= 0.05:
        return data, 0  # Original data is stationary
    
    # Step 2: If not stationary, try first difference
    # First difference = Current value - Previous value
    diff1 = data.diff().dropna()
    if adfuller(diff1)[1] <= 0.05:
        return diff1, 1  # First difference is stationary
    
    # Step 3: If still not stationary, use second difference
    # Second difference = First difference of first difference
    diff2 = diff1.diff().dropna()
    return diff2, 2  # Return second difference and order of differencing

def train_arima_model(data, d_value):
    """Train ARIMA model with simple parameter selection.
    
    ARIMA parameters:
    p: Order of AR (Auto Regressive) term - number of lags
    d: Order of differencing - already determined in make_stationary()
    q: Order of MA (Moving Average) term - number of lagged forecast errors
    """
    # Split data into training (80%) and testing (20%) sets
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # Initialize variables for best model selection
    best_aic = float('inf')  # Akaike Information Criterion - lower is better
    best_model = None
    best_order = None
    
    # Grid search for best p and q values
    # Keeping range small (0-2) for simplicity and efficiency
    for p in range(3):  # AR parameter
        for q in range(3):  # MA parameter
            try:
                # Try ARIMA with current p, d, q values
                model = ARIMA(train_data, order=(p, d_value, q))
                results = model.fit()
                
                # Update best model if current AIC is lower
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_model = results
                    best_order = (p, d_value, q)
                    
                print(f"ARIMA({p},{d_value},{q}) AIC: {results.aic:.2f}")
            except:
                # Skip if model fails to converge
                continue
    
    return best_model, best_order, train_data, test_data

def evaluate_and_plot(model, order, train_data, test_data):
    """Evaluate model performance and visualize predictions."""
    # Generate predictions for test period
    predictions = model.predict(
        start=len(train_data),  # Start from end of training data
        end=len(train_data) + len(test_data) - 1  # End at last test point
    )
    
    # Calculate Root Mean Square Error
    # RMSE measures the average magnitude of prediction errors
    rmse = np.sqrt(mean_squared_error(test_data, predictions))
    print(f"\nModel ARIMA{order} RMSE: {rmse:.2f}")
    
    # Plot actual vs predicted values
    plt.figure(figsize=(12, 6))
    plt.plot(test_data.index, test_data, label='Actual')
    plt.plot(test_data.index, predictions, color='red', label='Predicted')
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return predictions

def make_forecast(model, data, days=30):
    """Generate and visualize future price predictions.
    
    Args:
        model: Trained ARIMA model
        data: Original time series data
        days: Number of days to forecast (default: 30)
    """
    # Generate future predictions
    forecast = model.forecast(steps=days)
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    # Plot last 90 days of historical data for context
    plt.plot(data.index[-90:], data[-90:], label='Historical')
    # Plot forecast values
    plt.plot(
        pd.date_range(start=data.index[-1], periods=days+1)[1:],
        forecast,
        color='red',
        label='Forecast'
    )
    plt.title('Stock Price Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return forecast

def main():
    """Main execution function."""
    try:
        # Step 1: Load and prepare the data
        print("Loading data...")
        prices = load_data('sp500_data.csv')
        
        # Step 2: Transform data to achieve stationarity
        print("\nPreparing data...")
        stationary_data, d_value = make_stationary(prices)
        
        # Step 3: Train the ARIMA model
        print("\nTraining model...")
        model, order, train_data, test_data = train_arima_model(prices, d_value)
        
        if model is None:
            print("Could not find a suitable model.")
            return
        
        # Step 4: Evaluate model performance
        print("\nEvaluating model...")
        evaluate_and_plot(model, order, train_data, test_data)
        
        # Step 5: Generate future predictions
        print("\nGenerating forecast...")
        forecast = make_forecast(model, prices)
        
        print("\nAnalysis complete!")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")

if __name__ == "__main__":
    main()