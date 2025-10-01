"""
Data acquisition module for cryptocurrency trading data.
Fetches historical hourly data from CryptoCompare API.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataFetcher:
    """
    Fetches historical cryptocurrency data from CryptoCompare API.
    Provides hourly OHLCV data for training the DRL model.
    """

    def __init__(self, api_key=None):
        """
        Initialize the data fetcher.

        Args:
            api_key (str): Optional CryptoCompare API key for higher rate limits
        """
        self.api_key = api_key
        self.base_url = "https://min-api.cryptocompare.com/data/v2"
        self.rate_limit_delay = 1  # seconds between requests

    def get_historical_hourly(self, fsym='BTC', tsym='USD', limit=2000, toTs=None):
        """
        Fetch hourly historical data for a cryptocurrency pair.

        Args:
            fsym (str): From symbol (e.g., 'BTC')
            tsym (str): To symbol (e.g., 'USD')
            limit (int): Number of data points (max 2000)
            toTs (int): End timestamp (Unix timestamp)

        Returns:
            pd.DataFrame: DataFrame with OHLCV data
        """
        endpoint = "/histohour"
        url = self.base_url + endpoint

        params = {
            'fsym': fsym,
            'tsym': tsym,
            'limit': limit,
            'aggregate': 1  # 1 hour intervals
        }

        if toTs:
            params['toTs'] = toTs

        if self.api_key:
            params['api_key'] = self.api_key

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data['Response'] == 'Success':
                df = pd.DataFrame(data['Data']['Data'])

                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('timestamp', inplace=True)

                # Select and rename relevant columns
                df = df[['open', 'high', 'low', 'close', 'volumefrom', 'volumeto']]
                df.columns = ['open', 'high', 'low', 'close', 'volume_from', 'volume_to']

                # Convert to numeric types
                numeric_cols = ['open', 'high', 'low', 'close', 'volume_from', 'volume_to']
                df[numeric_cols] = df[numeric_cols].astype(float)

                # Remove zero volume entries
                df = df[df['volume_from'] > 0]

                logger.info(f"Successfully fetched {len(df)} hours of {fsym}/{tsym} data")
                return df
            else:
                logger.error(f"API Error: {data.get('Message', 'Unknown error')}")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None

    def get_multiple_days(self, fsym='BTC', tsym='USD', days=365):
        """
        Fetch data for multiple days by making multiple API requests.

        Args:
            fsym (str): From symbol
            tsym (str): To symbol
            days (int): Number of days of data to fetch

        Returns:
            pd.DataFrame: Combined DataFrame with all data
        """
        all_data = []
        current_ts = int(datetime.now().timestamp())

        # Calculate number of requests needed (2000 hours ≈ 83 days)
        hours_needed = days * 24
        requests_needed = int(np.ceil(hours_needed / 2000))

        logger.info(f"Fetching {days} days of data in {requests_needed} requests")

        for i in range(requests_needed):
            logger.info(f"Fetching batch {i+1}/{requests_needed}")

            data = self.get_historical_hourly(
                fsym=fsym,
                tsym=tsym,
                limit=2000,
                toTs=current_ts
            )

            if data is not None and not data.empty:
                all_data.append(data)
                # Update timestamp for next batch
                current_ts = int(data.index[0].timestamp())
            else:
                logger.warning(f"Failed to fetch batch {i+1}")
                break

            # Rate limiting
            if i < requests_needed - 1:  # Don't delay after last request
                time.sleep(self.rate_limit_delay)

        if all_data:
            # Combine all dataframes
            combined_df = pd.concat(all_data)

            # Remove duplicates and sort
            combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
            combined_df = combined_df.sort_index()

            logger.info(f"Successfully fetched {len(combined_df)} total data points")
            return combined_df

        logger.error("Failed to fetch any data")
        return None

    def fetch_daily_data(self, symbol='BTC', days=365):
        """
        Fetch daily data for the specified symbol and number of days.
        This is a convenience method that wraps get_multiple_days.
        
        Args:
            symbol (str): Cryptocurrency symbol (e.g., 'BTC', 'ETH')
            days (int): Number of days of data to fetch
            
        Returns:
            pd.DataFrame: DataFrame with daily OHLCV data
        """
        # Convert hourly data to daily by resampling
        hourly_data = self.get_multiple_days(fsym=symbol, tsym='USD', days=days)
        
        if hourly_data is None or len(hourly_data) == 0:
            return None
            
        # Resample to daily data
        daily_data = hourly_data.resample('D').agg({
            'open': 'first',
            'high': 'max', 
            'low': 'min',
            'close': 'last',
            'volume_from': 'sum',
            'volume_to': 'sum'
        }).dropna()
        
        # Rename volume column for consistency
        daily_data['volume'] = daily_data['volume_from']
        daily_data = daily_data[['open', 'high', 'low', 'close', 'volume']]
        
        logger.info(f"Converted to {len(daily_data)} days of {symbol} daily data")
        return daily_data


        """
        Save DataFrame to CSV file.

        Args:
            df (pd.DataFrame): Data to save
            filename (str): Filename without extension
            data_dir (str): Directory to save in
        """
        filepath = f"{data_dir}/{filename}.csv"
        df.to_csv(filepath)
        logger.info(f"Data saved to {filepath}")

    def load_data(self, filename, data_dir='data'):
        """
        Load DataFrame from CSV file.

        Args:
            filename (str): Filename without extension
            data_dir (str): Directory to load from

        Returns:
            pd.DataFrame: Loaded data
        """
        filepath = f"{data_dir}/{filename}.csv"
        try:
            df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
            logger.info(f"Data loaded from {filepath}")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            return None


# Example usage
if __name__ == "__main__":
    # Initialize fetcher (add your API key if you have one)
    fetcher = DataFetcher()

    # Fetch Bitcoin data for 30 days
    btc_data = fetcher.get_multiple_days('BTC', 'USD', days=30)

    if btc_data is not None:
        print("Sample of fetched data:")
        print(btc_data.head())
        print(f"\nData shape: {btc_data.shape}")
        print(f"Date range: {btc_data.index.min()} to {btc_data.index.max()}")

        # Save data
        fetcher.save_data(btc_data, 'btc_usd_30d')