from data_processor import TimeSeriesProcessor
from analysis import TimeSeriesAnalyzer
from datetime import datetime

def main():
    # Create processor and load S&P 500 data
    processor = TimeSeriesProcessor(
        ticker='^GSPC',  # S&P 500 index
        start_date='2019-01-01',  # Last 5 years
        end_date=datetime.now().strftime('%Y-%m-%d')
    )
    
    # Load and clean data
    data = processor.load_data()
    cleaned_data = processor.clean_data()
    
    # Create analyzer and perform analysis
    analyzer = TimeSeriesAnalyzer(processor)
    analyzer.basic_analysis()

if __name__ == "__main__":
    main() 