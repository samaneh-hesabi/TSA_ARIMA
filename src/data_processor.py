import pandas as pd
import numpy as np
import logging
import yfinance as yf
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TimeSeriesProcessor:
    def __init__(self, ticker='^GSPC', start_date=None, end_date=None):
        """
        Initialize the TimeSeriesProcessor with financial data parameters.
        
        Args:
            ticker (str): Stock ticker symbol (default: S&P 500)
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
        """
        self.ticker = ticker
        self.start_date = start_date or (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.data = None
        self.processed_data = None
        
    def load_data(self):
        """
        Load financial data from Yahoo Finance.
        
        Returns:
            pandas.DataFrame: Loaded data
        """
        try:
            # Download data from Yahoo Finance
            stock = yf.Ticker(self.ticker)
            self.data = stock.history(start=self.start_date, end=self.end_date)
            
            # Keep only relevant columns
            self.data = self.data[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Calculate daily returns
            self.data['Returns'] = self.data['Close'].pct_change()
            
            # Calculate log returns (better for financial analysis)
            self.data['Log_Returns'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
            
            logger.info(f"Successfully loaded {self.ticker} data from {self.start_date} to {self.end_date}")
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def clean_data(self):
        """
        Clean the loaded financial data by:
        1. Handling missing values
        2. Calculating additional features
        3. Setting proper index
        
        Returns:
            pandas.DataFrame: Cleaned data
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        try:
            # Create a copy of the data
            self.processed_data = self.data.copy()
            
            # Handle missing values
            self.processed_data = self.processed_data.fillna(method='ffill')
            
            # Calculate moving averages
            self.processed_data['MA20'] = self.processed_data['Close'].rolling(window=20).mean()
            self.processed_data['MA50'] = self.processed_data['Close'].rolling(window=50).mean()
            
            # Calculate volatility
            self.processed_data['Volatility'] = self.processed_data['Returns'].rolling(window=20).std() * np.sqrt(252)
            
            logger.info("Data cleaning completed successfully")
            return self.processed_data
            
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            raise 