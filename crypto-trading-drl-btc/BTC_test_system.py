"""
Bitcoin DRL Trading System Test Suite
Comprehensive testing script for the Bitcoin trading system components.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_fetcher import DataFetcher
from trading_env import TradingEnvironment
from neural_networks import PPOAgent
from technical_indicators import TechnicalIndicators

class TestBitcoinDataFetcher(unittest.TestCase):
    """Test Bitcoin data fetching functionality."""
    
    def setUp(self):
        self.data_fetcher = DataFetcher()
    
    def test_fetch_bitcoin_data(self):
        """Test Bitcoin data fetching."""
        print("Testing Bitcoin data fetching...")
        
        # Test with small dataset
        data = self.data_fetcher.fetch_daily_data('BTC', days=30)
        
        self.assertIsNotNone(data, "Data should not be None")
        self.assertGreater(len(data), 0, "Data should not be empty")
        self.assertIn('close', data.columns, "Data should contain 'close' column")
        self.assertIn('volume', data.columns, "Data should contain 'volume' column")
        
        print(f"✓ Successfully fetched {len(data)} days of Bitcoin data")
        print(f"✓ Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")

class TestBitcoinTradingEnvironment(unittest.TestCase):
    """Test Bitcoin trading environment."""
    
    def setUp(self):
        # Create sample Bitcoin data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        # Simulate realistic Bitcoin price movement
        initial_price = 30000
        returns = np.random.normal(0.001, 0.03, 100)  # Daily returns
        prices = [initial_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        self.test_data = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.uniform(1000000, 5000000, 100)
        }, index=dates)
        
        self.env = TradingEnvironment(
            data=self.test_data,
            initial_balance=10000,
            transaction_fee=0.001,
            lookback_window=10,
            symbol='BTC'
        )
    
    def test_environment_initialization(self):
        """Test environment initialization."""
        print("Testing Bitcoin trading environment initialization...")
        
        self.assertEqual(self.env.initial_balance, 10000)
        self.assertEqual(self.env.current_balance, 10000)
        self.assertEqual(self.env.current_holdings, 0)
        self.assertEqual(self.env.transaction_fee, 0.001)
        
        print("✓ Environment initialized correctly")
    
    def test_environment_reset(self):
        """Test environment reset functionality."""
        print("Testing environment reset...")
        
        state = self.env.reset()
        self.assertIsInstance(state, np.ndarray)
        self.assertGreater(len(state), 0)
        
        print(f"✓ Environment reset successful, state shape: {state.shape}")
    
    def test_environment_actions(self):
        """Test all three actions in the environment."""
        print("Testing Bitcoin trading actions...")
        
        self.env.reset()
        
        # Test Hold action
        initial_balance = self.env.current_balance
        state, reward, done, info = self.env.step(0)  # Hold
        self.assertEqual(self.env.current_balance, initial_balance)
        print("✓ Hold action works correctly")
        
        # Test Buy action
        self.env.reset()
        state, reward, done, info = self.env.step(1)  # Buy
        self.assertLess(self.env.current_balance, self.env.initial_balance)
        self.assertGreater(self.env.current_holdings, 0)
        print("✓ Buy action works correctly")
        
        # Test Sell action (need holdings first)
        self.env.reset()
        self.env.step(1)  # Buy first
        holdings_before = self.env.current_holdings
        self.env.step(2)  # Sell
        self.assertLess(self.env.current_holdings, holdings_before)
        print("✓ Sell action works correctly")

class TestBitcoinTechnicalIndicators(unittest.TestCase):
    """Test technical indicators for Bitcoin."""
    
    def setUp(self):
        # Create sample Bitcoin price data
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        
        prices = [30000]
        for _ in range(49):
            prices.append(prices[-1] * (1 + np.random.normal(0, 0.02)))
        
        self.data = pd.DataFrame({
            'open': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': np.random.uniform(1000000, 3000000, 50)
        }, index=dates)
        
        self.ti = TechnicalIndicators()
    
    def test_rsi_calculation(self):
        """Test RSI calculation for Bitcoin."""
        print("Testing RSI calculation for Bitcoin...")
        
        rsi = self.ti.calculate_rsi(self.data['close'])
        
        self.assertIsInstance(rsi, pd.Series)
        self.assertTrue(all(0 <= val <= 100 for val in rsi.dropna()))
        
        print(f"✓ RSI calculated successfully, range: {rsi.min():.2f} - {rsi.max():.2f}")
    
    def test_atr_calculation(self):
        """Test ATR calculation for Bitcoin."""
        print("Testing ATR calculation for Bitcoin...")
        
        atr = self.ti.calculate_atr(self.data)
        
        self.assertIsInstance(atr, pd.Series)
        self.assertTrue(all(val >= 0 for val in atr.dropna()))
        
        print(f"✓ ATR calculated successfully, range: {atr.min():.2f} - {atr.max():.2f}")
    
    def test_obv_calculation(self):
        """Test OBV calculation for Bitcoin."""
        print("Testing OBV calculation for Bitcoin...")
        
        obv = self.ti.calculate_obv(self.data['close'], self.data['volume'])
        
        self.assertIsInstance(obv, pd.Series)
        self.assertEqual(len(obv), len(self.data))
        
        print(f"✓ OBV calculated successfully, range: {obv.min():.0f} - {obv.max():.0f}")

class TestBitcoinPPOAgent(unittest.TestCase):
    """Test PPO agent for Bitcoin trading."""
    
    def setUp(self):
        self.state_size = 50
        self.action_size = 3
        self.agent = PPOAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            lr=0.001
        )
    
    def test_agent_initialization(self):
        """Test PPO agent initialization."""
        print("Testing Bitcoin PPO agent initialization...")
        
        self.assertEqual(self.agent.state_size, self.state_size)
        self.assertEqual(self.agent.action_size, self.action_size)
        self.assertIsNotNone(self.agent.actor)
        self.assertIsNotNone(self.agent.critic)
        
        print("✓ PPO agent initialized correctly")
    
    def test_agent_action_selection(self):
        """Test agent action selection."""
        print("Testing agent action selection...")
        
        state = np.random.random(self.state_size)
        action, log_prob = self.agent.act(state)
        
        self.assertIn(action, [0, 1, 2])
        self.assertIsInstance(log_prob, float)
        
        print(f"✓ Agent selected action: {action} with log_prob: {log_prob:.4f}")
    
    def test_agent_training_step(self):
        """Test agent training step."""
        print("Testing agent training step...")
        
        # Add some dummy experiences
        for _ in range(10):
            state = np.random.random(self.state_size)
            action = np.random.choice([0, 1, 2])
            log_prob = np.random.random()
            reward = np.random.normal(0, 1)
            done = False
            
            self.agent.store_transition(state, action, log_prob, reward, done)
        
        # Test update
        try:
            self.agent.update()
            print("✓ Agent training step completed successfully")
        except Exception as e:
            self.fail(f"Agent update failed: {str(e)}")

def run_bitcoin_system_tests():
    """Run comprehensive Bitcoin system tests."""
    print("="*60)
    print("BITCOIN DRL TRADING SYSTEM - COMPREHENSIVE TESTS")
    print("="*60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestBitcoinDataFetcher,
        TestBitcoinTradingEnvironment,
        TestBitcoinTechnicalIndicators,
        TestBitcoinPPOAgent
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    print("\n" + "="*60)
    print("BITCOIN SYSTEM TESTS SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    if len(result.failures) == 0 and len(result.errors) == 0:
        print("\n🎉 ALL BITCOIN SYSTEM TESTS PASSED!")
        print("The Bitcoin DRL trading system is ready for training and backtesting.")
    else:
        print("\n❌ Some tests failed. Please review and fix issues before proceeding.")
    
    return result

if __name__ == "__main__":
    try:
        result = run_bitcoin_system_tests()
        
        # Additional system readiness check
        print("\n" + "="*60)
        print("BITCOIN SYSTEM READINESS CHECK")
        print("="*60)
        
        # Check required directories
        required_dirs = ['src', 'data', 'full_training_results']
        for dir_name in required_dirs:
            if os.path.exists(dir_name):
                print(f"✓ Directory '{dir_name}' exists")
            else:
                print(f"✗ Directory '{dir_name}' missing")
                os.makedirs(dir_name, exist_ok=True)
                print(f"✓ Created directory '{dir_name}'")
        
        # Check required files
        required_files = [
            'src/data_fetcher.py',
            'src/trading_env.py',
            'src/neural_networks.py',
            'src/technical_indicators.py',
            'train_bitcoin_agent.py',
            'backtest_bitcoin_agent.py'
        ]
        
        missing_files = []
        for file_name in required_files:
            if os.path.exists(file_name):
                print(f"✓ File '{file_name}' exists")
            else:
                print(f"✗ File '{file_name}' missing")
                missing_files.append(file_name)
        
        if not missing_files and len(result.failures) == 0 and len(result.errors) == 0:
            print(f"\n🚀 BITCOIN SYSTEM IS READY!")
            print("You can now run:")
            print("1. python train_bitcoin_agent.py  # Train the agent")
            print("2. python backtest_bitcoin_agent.py  # Run backtest analysis")
        else:
            print(f"\n⚠️  System not fully ready. Address the issues above first.")
    
    except Exception as e:
        print(f"Test execution failed: {str(e)}")
        raise