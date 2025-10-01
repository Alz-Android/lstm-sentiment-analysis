"""
Custom RL environment for cryptocurrency trading.
Based on the paper's environment design with buy/hold/sell actions.
"""

import gym
import numpy as np
import pandas as pd
from gym import spaces
import logging

logger = logging.getLogger(__name__)

class TradingEnvironment(gym.Env):
    """
    Custom OpenAI Gym environment for cryptocurrency trading.
    The agent can take three actions: Buy, Hold, Sell.
    """

    def __init__(self, df, initial_balance=10000, lookback_window=100):
        """
        Initialize the trading environment.

        Args:
            df (pd.DataFrame): DataFrame with OHLCV and technical indicators
            initial_balance (float): Starting cash balance
            lookback_window (int): Number of historical timesteps in state
        """
        super(TradingEnvironment, self).__init__()

        self.df = df.copy()
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window

        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)

        # State space: lookback_window timesteps × 4 features (close, rsi, atr, obv)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(lookback_window, 4),
            dtype=np.float32
        )

        # Environment state
        self.current_step = lookback_window
        self.balance = initial_balance
        self.crypto_held = 0
        self.net_worth = initial_balance
        self.prev_net_worth = initial_balance

        # Transaction costs
        self.transaction_fee = 0.001  # 0.1%

        logger.info(f"Environment initialized with {len(df)} data points")
        logger.info(f"Initial balance: ${initial_balance}")
        logger.info(f"Lookback window: {lookback_window}")
    
    @property
    def current_balance(self):
        """Current cash balance (compatibility property)"""
        return self.balance
    
    @property
    def current_holdings(self):
        """Current crypto holdings (compatibility property)"""
        return self.crypto_held
    
    @property
    def current_price(self):
        """Current price (compatibility property)"""
        return self.df.iloc[self.current_step]['close']
    
    @property
    def current_date(self):
        """Current date (compatibility property)"""
        return self.df.index[self.current_step]

    def reset(self):
        """
        Reset the environment to initial state.

        Returns:
            np.array: Initial state observation
        """
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.crypto_held = 0
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance

        return self._get_observation()

    def step(self, action):
        """
        Execute one step in the environment.

        Args:
            action (int): Action to take (0=Hold, 1=Buy, 2=Sell)

        Returns:
            tuple: (observation, reward, done, info)
        """
        # Store previous net worth
        self.prev_net_worth = self.net_worth

        # Get current price
        current_price = self.df.iloc[self.current_step]['close']

        # Execute action
        if action == 1:  # Buy
            self._buy_crypto(current_price)
        elif action == 2:  # Sell
            self._sell_crypto(current_price)
        # Action 0 = Hold (do nothing)

        # Update net worth
        self.net_worth = self.balance + (self.crypto_held * current_price)

        # Calculate reward (change in net worth)
        reward = self.net_worth - self.prev_net_worth

        # Move to next step
        self.current_step += 1

        # Check if episode is done
        done = self.current_step >= len(self.df) - 1

        # Get next observation
        obs = self._get_observation()

        # Additional info
        info = {
            'net_worth': self.net_worth,
            'balance': self.balance,
            'crypto_held': self.crypto_held,
            'current_price': current_price,
            'step': self.current_step
        }

        return obs, reward, done, info

    def _buy_crypto(self, price):
        """Execute buy action"""
        if self.balance > 0:
            # Calculate amount to buy (all available balance)
            crypto_to_buy = self.balance / price
            # Apply transaction fee
            crypto_to_buy *= (1 - self.transaction_fee)

            self.crypto_held += crypto_to_buy
            self.balance = 0

            logger.debug(f"Bought {crypto_to_buy:.6f} crypto at ${price:.2f}")

    def _sell_crypto(self, price):
        """Execute sell action"""
        if self.crypto_held > 0:
            # Calculate amount to receive
            cash_to_receive = self.crypto_held * price
            # Apply transaction fee
            cash_to_receive *= (1 - self.transaction_fee)

            self.balance += cash_to_receive
            self.crypto_held = 0

            logger.debug(f"Sold crypto at ${price:.2f}, received ${cash_to_receive:.2f}")

    def _get_observation(self):
        """
        Get current state observation.

        Returns:
            np.array: State observation with shape (lookback_window, 4)
        """
        # Get lookback window data
        start_idx = self.current_step - self.lookback_window
        end_idx = self.current_step

        # Select features: close, rsi, atr, obv
        features = ['close', 'rsi', 'atr', 'obv']
        obs_data = self.df[features].iloc[start_idx:end_idx].values

        return obs_data.astype(np.float32)

    def render(self, mode='human'):
        """Render environment state"""
        print(f"Step: {self.current_step}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Crypto Held: {self.crypto_held:.6f}")
        print(f"Net Worth: ${self.net_worth:.2f}")
        print(f"Current Price: ${self.df.iloc[self.current_step]['close']:.2f}")


# Example usage and testing
if __name__ == "__main__":
    import sys
    import os

    # Add src to path
    sys.path.append(os.path.dirname(__file__))

    from data_fetcher import DataFetcher
    from technical_indicators import TechnicalIndicators

    def test_trading_env():
        """Test the trading environment"""
        print("Testing Crypto Trading Environment...")

        # Load data with indicators
        fetcher = DataFetcher()
        df = fetcher.load_data('btc_usd_with_indicators')

        if df is None or len(df) < 110:  # Need at least 110 points for 100 lookback + 10 steps
            print("Not enough data. Fetching more...")
            df = fetcher.get_historical_hourly('BTC', 'USD', limit=150)
            if df is not None:
                ti = TechnicalIndicators()
                df = ti.add_technical_indicators(df)
                fetcher.save_data(df, 'btc_usd_with_indicators')

        if df is None or len(df) < 110:
            print("Still not enough data for testing")
            return

        print(f"Data shape: {df.shape}")

        # Create environment
        env = TradingEnvironment(df, initial_balance=10000, lookback_window=100)

        # Reset environment
        obs = env.reset()
        print(f"Initial observation shape: {obs.shape}")

        # Test a few steps
        total_reward = 0
        for i in range(10):
            action = np.random.randint(0, 3)  # Random action
            obs, reward, done, info = env.step(action)

            total_reward += reward
            print(f"Step {i+1}:")
            print(f"  Action: {['Hold', 'Buy', 'Sell'][action]}")
            print(f"  Reward: ${reward:.2f}")
            print(f"  Net Worth: ${info['net_worth']:.2f}")
            print(f"  Crypto Held: {info['crypto_held']:.6f}")

            if done:
                break

        print(f"\nTotal reward: ${total_reward:.2f}")
        print("Environment test completed!")

    test_trading_env()