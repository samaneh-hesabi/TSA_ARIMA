import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import warnings
import os

warnings.filterwarnings("ignore")

# 1. Load the dataset
df = pd.read_csv('scr2/btc-usd-max.csv')

# 2. Preprocessing
df = df.rename(columns=lambda x: x.strip())
df['snapped_at'] = pd.to_datetime(df['snapped_at'])
df = df.sort_values('snapped_at')
df.set_index('snapped_at', inplace=True)

# Ensure 'price' column is numeric
df['price'] = df['price'].astype(str).str.replace(',', '').astype(float)

# Display the first few rows
print("Sample data:")
print(df.head())

# 3. Augmented Dickey-Fuller Test for stationarity
result = adfuller(df['price'])
print('\nAugmented Dickey-Fuller Test:')
print(f'ADF Statistic: {result[0]:.4f}')
print(f'p-value: {result[1]:.4f}')
if result[1] < 0.05:
    print("The time series is stationary.")
else:
    print("The time series is not stationary. Differencing is needed.")

# 4. ACF and PACF plots
plt.figure(figsize=(12,5))
plt.subplot(121)
plot_acf(df['price'], ax=plt.gca(), lags=40)
plt.subplot(122)
plot_pacf(df['price'], ax=plt.gca(), lags=40)
plt.tight_layout()
plt.show()

# 5. Fit ARIMA model (assuming stationarity, so d=0)
model = ARIMA(df['price'], order=(1, 0, 1))  # you may tune (p,d,q)
result = model.fit()
print('\n ARIMA Model Summary:')
print(result.summary())

# 6. Analyze residuals
residuals = result.resid
plt.figure(figsize=(10,4))
plt.plot(residuals)
plt.title("Model Residuals")
plt.show()

plot_acf(residuals, lags=40)
plt.title("ACF of Residuals")
plt.show()

# 7. Forecast the next 10 days
forecast = result.forecast(steps=10)
print("\n🔮 10-Day Forecast:")
print(forecast)

# Plot forecast
forecast.plot(title="Bitcoin Price Forecast (next 10 steps)")
plt.ylabel("Price (USD)")
plt.xlabel("Forecast Index")
plt.grid(True)
plt.show()
