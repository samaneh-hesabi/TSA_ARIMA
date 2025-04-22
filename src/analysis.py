import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TimeSeriesAnalyzer:
    def __init__(self, processor):
        """
        Initialize the TimeSeriesAnalyzer with a TimeSeriesProcessor instance.
        
        Args:
            processor (TimeSeriesProcessor): An instance of TimeSeriesProcessor
        """
        self.processor = processor
        
    def basic_analysis(self):
        """
        Perform comprehensive analysis on the processed data:
        1. Display basic statistics
        2. Check for stationarity
        3. Analyze autocorrelation
        4. Test for normality
        5. Visualize data distributions
        """
        if self.processor.processed_data is None:
            raise ValueError("No processed data available. Please clean data first.")
            
        try:
            # 1. Basic Dataset Information
            print("\n=== Dataset Information ===")
            print(f"Time Period: {self.processor.processed_data.index[0]} to {self.processor.processed_data.index[-1]}")
            print(f"Total Trading Days: {len(self.processor.processed_data)}")
            print(f"Missing Values: {self.processor.processed_data.isnull().sum().sum()}")
            
            # 2. Basic Statistics
            print("\n=== Basic Statistics ===")
            print("\nClosing Prices:")
            print(self.processor.processed_data['Close'].describe())
            print("\nDaily Returns:")
            print(self.processor.processed_data['Returns'].describe())
            print("\nLog Returns:")
            print(self.processor.processed_data['Log_Returns'].describe())
            
            # 3. Stationarity Tests
            print("\n=== Stationarity Tests ===")
            # ADF test for price
            adf_result = adfuller(self.processor.processed_data['Close'].dropna())
            print(f"ADF Test for Price - p-value: {adf_result[1]:.4f}")
            
            # ADF test for returns
            adf_result = adfuller(self.processor.processed_data['Returns'].dropna())
            print(f"ADF Test for Returns - p-value: {adf_result[1]:.4f}")
            
            # 4. Normality Tests
            print("\n=== Normality Tests ===")
            # Shapiro-Wilk test for returns
            shapiro_test = stats.shapiro(self.processor.processed_data['Returns'].dropna())
            print(f"Shapiro-Wilk Test - p-value: {shapiro_test[1]:.4f}")
            
            # 5. Autocorrelation Analysis
            print("\n=== Autocorrelation Analysis ===")
            # Plot ACF and PACF
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            plot_acf(self.processor.processed_data['Returns'].dropna(), ax=ax1, lags=40)
            plot_pacf(self.processor.processed_data['Returns'].dropna(), ax=ax2, lags=40)
            plt.tight_layout()
            plt.show()
            
            # 6. Time Series Plots
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
            
            # Price with moving averages
            ax1.plot(self.processor.processed_data.index, self.processor.processed_data['Close'], label='Close Price')
            ax1.plot(self.processor.processed_data.index, self.processor.processed_data['MA20'], label='20-day MA')
            ax1.plot(self.processor.processed_data.index, self.processor.processed_data['MA50'], label='50-day MA')
            ax1.set_title('S&P 500 Price with Moving Averages')
            ax1.set_ylabel('Price')
            ax1.legend()
            ax1.grid(True)
            
            # Returns
            ax2.plot(self.processor.processed_data.index, self.processor.processed_data['Returns'])
            ax2.set_title('Daily Returns')
            ax2.set_ylabel('Returns')
            ax2.grid(True)
            
            # Volatility
            ax3.plot(self.processor.processed_data.index, self.processor.processed_data['Volatility'])
            ax3.set_title('20-day Rolling Volatility')
            ax3.set_ylabel('Volatility')
            ax3.grid(True)
            
            plt.tight_layout()
            plt.show()
            
            # 7. Distribution Analysis
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Returns distribution
            sns.histplot(self.processor.processed_data['Returns'].dropna(), kde=True, ax=ax1)
            ax1.set_title('Distribution of Daily Returns')
            ax1.set_xlabel('Returns')
            ax1.set_ylabel('Frequency')
            ax1.grid(True)
            
            # QQ plot for normality check
            stats.probplot(self.processor.processed_data['Returns'].dropna(), dist="norm", plot=ax2)
            ax2.set_title('Q-Q Plot for Returns')
            ax2.grid(True)
            
            plt.tight_layout()
            plt.show()
            
            # 8. Volume Analysis
            print("\n=== Volume Analysis ===")
            print("\nVolume Statistics:")
            print(self.processor.processed_data['Volume'].describe())
            
            # Plot volume
            plt.figure(figsize=(15, 6))
            plt.plot(self.processor.processed_data.index, self.processor.processed_data['Volume'])
            plt.title('Trading Volume')
            plt.xlabel('Date')
            plt.ylabel('Volume')
            plt.grid(True)
            plt.show()
            
            logger.info("Comprehensive analysis completed")
            
        except Exception as e:
            logger.error(f"Error in analysis: {str(e)}")
            raise 