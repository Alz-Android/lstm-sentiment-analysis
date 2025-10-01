"""
Technical indicators module for cryptocurrency trading.
Calculates RSI, ATR, and OBV indicators as used in the DRL paper.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """
    Calculates technical indicators for cryptocurrency price data.
    Based on the indicators used in the DRL paper: RSI, ATR, OBV.
    """

    def __init__(self):
        """Initialize the technical indicators calculator"""
        pass

    def calculate_rsi(self, prices, period=14):
        """
        Calculate Relative Strength Index (RSI)

        Args:
            prices (pd.Series): Price series (typically close prices)
            period (int): Period for RSI calculation (default 14)

        Returns:
            pd.Series: RSI values
        """
        try:
            # Calculate price changes
            delta = prices.diff()

            # Separate gains and losses
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            # Calculate RS (Relative Strength)
            rs = gain / loss

            # Calculate RSI
            rsi = 100 - (100 / (1 + rs))

            logger.info(f"Calculated RSI with period {period}")
            return rsi

        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return pd.Series(index=prices.index, dtype=float)

    def calculate_atr(self, high, low, close, period=14):
        """
        Calculate Average True Range (ATR)

        Args:
            high (pd.Series): High prices
            low (pd.Series): Low prices
            close (pd.Series): Close prices
            period (int): Period for ATR calculation (default 14)

        Returns:
            pd.Series: ATR values
        """
        try:
            # Calculate True Range components
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))

            # True Range is the maximum of the three
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # Calculate ATR as rolling mean of True Range
            atr = tr.rolling(window=period).mean()

            logger.info(f"Calculated ATR with period {period}")
            return atr

        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return pd.Series(index=high.index, dtype=float)

    def calculate_obv(self, close, volume):
        """
        Calculate On-Balance Volume (OBV)

        Args:
            close (pd.Series): Close prices
            volume (pd.Series): Volume data

        Returns:
            pd.Series: OBV values
        """
        try:
            # Calculate price changes
            price_change = close.diff()

            # Initialize OBV
            obv = pd.Series(index=close.index, dtype=float)
            obv.iloc[0] = volume.iloc[0]

            # Calculate OBV
            for i in range(1, len(close)):
                if price_change.iloc[i] > 0:
                    obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
                elif price_change.iloc[i] < 0:
                    obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]

            logger.info("Calculated OBV")
            return obv

        except Exception as e:
            logger.error(f"Error calculating OBV: {e}")
            return pd.Series(index=close.index, dtype=float)

    def add_technical_indicators(self, df, rsi_period=14, atr_period=14):
        """
        Add all technical indicators to the DataFrame

        Args:
            df (pd.DataFrame): OHLCV DataFrame
            rsi_period (int): Period for RSI
            atr_period (int): Period for ATR

        Returns:
            pd.DataFrame: DataFrame with technical indicators added
        """
        try:
            df = df.copy()

            # Calculate RSI
            df['rsi'] = self.calculate_rsi(df['close'], rsi_period)

            # Calculate ATR
            df['atr'] = self.calculate_atr(df['high'], df['low'], df['close'], atr_period)

            # Calculate OBV - use volume_from for crypto data
            volume_col = 'volume_from' if 'volume_from' in df.columns else 'volume'
            df['obv'] = self.calculate_obv(df['close'], df[volume_col])

            # Fill NaN values with forward fill, then backward fill
            df = df.ffill().bfill()

            logger.info("Added all technical indicators to DataFrame")
            logger.info(f"DataFrame shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")

            return df

        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return df

    def normalize_features(self, df, feature_columns=None):
        """
        Normalize features using z-score normalization

        Args:
            df (pd.DataFrame): DataFrame with features
            feature_columns (list): Columns to normalize (default: all numeric)

        Returns:
            pd.DataFrame: DataFrame with normalized features
        """
        try:
            df = df.copy()

            if feature_columns is None:
                # Normalize all numeric columns except timestamp
                feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()

            for col in feature_columns:
                if col in df.columns:
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    if std_val != 0:
                        df[f'{col}_norm'] = (df[col] - mean_val) / std_val
                    else:
                        df[f'{col}_norm'] = 0

            logger.info(f"Normalized {len(feature_columns)} features")
            return df

        except Exception as e:
            logger.error(f"Error normalizing features: {e}")
            return df

    def prepare_features_for_model(self, df, lookback_window=100):
        """
        Prepare features for the DRL model with lookback window

        Args:
            df (pd.DataFrame): DataFrame with OHLCV and indicators
            lookback_window (int): Number of historical timesteps to include

        Returns:
            np.array: Feature array ready for model input
        """
        try:
            # Select features for the model
            feature_cols = ['close', 'rsi', 'atr', 'obv']

            # Ensure we have enough data
            if len(df) < lookback_window:
                logger.warning(f"Data length {len(df)} is less than lookback window {lookback_window}")
                return None

            # Create sequences
            features = []
            for i in range(lookback_window, len(df)):
                window_data = df[feature_cols].iloc[i-lookback_window:i].values
                features.append(window_data)

            features_array = np.array(features)
            logger.info(f"Prepared features with shape: {features_array.shape}")
            logger.info(f"Features: {feature_cols}")

            return features_array

        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None


# Example usage and testing
if __name__ == "__main__":
    import sys
    import os

    # Add src to path for imports
    sys.path.append(os.path.dirname(__file__))

    from data_fetcher import CryptoCompareDataFetcher

    def test_technical_indicators():
        """Test the technical indicators calculation"""
        print("Testing Technical Indicators...")

        # Load test data
        fetcher = CryptoCompareDataFetcher()
        df = fetcher.load_data('btc_usd_test')

        if df is None or df.empty:
            print("No test data found. Fetching new data...")
            df = fetcher.get_historical_hourly('BTC', 'USD', limit=50)
            if df is None:
                print("Failed to fetch data")
                return

        print(f"Original data shape: {df.shape}")
        print("Original columns:", list(df.columns))

        # Initialize technical indicators
        ti = TechnicalIndicators()

        # Add technical indicators
        df_with_indicators = ti.add_technical_indicators(df)

        print(f"Data with indicators shape: {df_with_indicators.shape}")
        print("New columns:", list(df_with_indicators.columns))

        # Show sample data
        print("\nSample data with indicators:")
        print(df_with_indicators[['close', 'rsi', 'atr', 'obv']].tail())

        # Test feature preparation
        features = ti.prepare_features_for_model(df_with_indicators)
        if features is not None:
            print(f"\nPrepared features shape: {features.shape}")
            print("Ready for model input!")

    test_technical_indicators()