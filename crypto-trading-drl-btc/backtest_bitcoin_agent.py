"""
Bitcoin DRL Trading Agent Backtest Script
Tests the trained PPO agent on unseen Bitcoin data with comprehensive analysis.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_fetcher import DataFetcher
from trading_env import TradingEnvironment
from neural_networks import PPOAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BitcoinBacktester:
    def __init__(self, model_path):
        """Initialize the Bitcoin backtester."""
        self.model_path = model_path
        self.data_fetcher = DataFetcher()
        
    def load_agent(self, state_size, action_size):
        """Load the trained PPO agent."""
        agent = PPOAgent(state_size, action_size)
        agent.load_model(self.model_path)
        return agent
    
    def get_test_data(self, days_back=365):
        """Get recent Bitcoin data for testing."""
        logger.info(f"Fetching {days_back} days of Bitcoin test data...")
        
        test_data = self.data_fetcher.fetch_daily_data(
            symbol='BTC',
            days=days_back
        )
        
        if test_data is None or len(test_data) < 30:
            raise ValueError("Insufficient test data")
            
        logger.info(f"Test data: {len(test_data)} days from {test_data.index[0]} to {test_data.index[-1]}")
        return test_data
    
    def run_backtest(self, test_data, strategy='stochastic', initial_balance=10000):
        """Run backtest with the trained agent."""
        logger.info(f"Running {strategy} backtest...")
        
        # Create test environment
        env = TradingEnvironment(
            df=test_data,
            initial_balance=initial_balance,
            lookback_window=30
        )
        
        # Load agent
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        agent = self.load_agent(state_size, action_size)
        
        # Run backtest
        state = env.reset()
        done = False
        
        # Track metrics
        portfolio_values = [initial_balance]
        actions_taken = []
        prices = []
        dates = []
        holdings = []
        cash_balances = []
        
        while not done:
            if strategy == 'stochastic':
                # Use stochastic action selection (sample from probability distribution)
                action, _ = agent.act(state)
            else:
                # Use deterministic action selection (highest probability)
                action_probs = agent.get_action_probabilities(state)
                action = np.argmax(action_probs)
            
            next_state, reward, done, info = env.step(action)
            
            # Record metrics
            current_portfolio_value = env.current_balance + (env.current_holdings * env.current_price)
            portfolio_values.append(current_portfolio_value)
            actions_taken.append(action)
            prices.append(env.current_price)
            dates.append(env.current_date)
            holdings.append(env.current_holdings)
            cash_balances.append(env.current_balance)
            
            state = next_state
        
        # Calculate final metrics
        final_value = portfolio_values[-1]
        total_return = (final_value / initial_balance - 1) * 100
        
        # Calculate buy and hold return
        initial_price = test_data['close'].iloc[30]  # After lookback window
        final_price = test_data['close'].iloc[-1]
        buy_hold_return = (final_price / initial_price - 1) * 100
        
        results = {
            'strategy': strategy,
            'initial_balance': initial_balance,
            'final_value': final_value,
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'outperformance': total_return - buy_hold_return,
            'portfolio_values': portfolio_values,
            'actions_taken': actions_taken,
            'prices': prices,
            'dates': dates,
            'holdings': holdings,
            'cash_balances': cash_balances,
            'test_period': f"{test_data.index[0].strftime('%Y-%m-%d')} to {test_data.index[-1].strftime('%Y-%m-%d')}"
        }
        
        return results
    
    def analyze_actions(self, actions_taken):
        """Analyze the distribution of actions taken."""
        action_names = ['Hold', 'Buy', 'Sell']
        action_counts = [actions_taken.count(i) for i in range(3)]
        action_percentages = [count/len(actions_taken)*100 for count in action_counts]
        
        return {
            'action_counts': action_counts,
            'action_percentages': action_percentages,
            'action_names': action_names
        }
    
    def save_results(self, results, output_file):
        """Save backtest results to CSV."""
        df_data = {
            'date': results['dates'],
            'price': results['prices'],
            'action': [['Hold', 'Buy', 'Sell'][a] for a in results['actions_taken']],
            'portfolio_value': results['portfolio_values'][1:],  # Skip initial value
            'holdings': results['holdings'],
            'cash_balance': results['cash_balances']
        }
        
        df = pd.DataFrame(df_data)
        df.to_csv(output_file, index=False)
        logger.info(f"Detailed results saved to {output_file}")
    
    def plot_results(self, results, save_path=None):
        """Plot backtest results."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Portfolio value over time
        ax1.plot(results['dates'], results['portfolio_values'][1:], label=f"{results['strategy'].title()} Strategy", linewidth=2)
        
        # Calculate buy and hold portfolio value
        initial_price = results['prices'][0]
        buy_hold_values = [results['initial_balance'] * (price / initial_price) for price in results['prices']]
        ax1.plot(results['dates'], buy_hold_values, label='Buy & Hold', linewidth=2, alpha=0.7)
        
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Bitcoin price
        ax2.plot(results['dates'], results['prices'], color='orange', linewidth=2)
        ax2.set_title('Bitcoin Price')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Price ($)')
        ax2.grid(True, alpha=0.3)
        
        # Actions distribution
        action_analysis = self.analyze_actions(results['actions_taken'])
        ax3.bar(action_analysis['action_names'], action_analysis['action_percentages'], 
                color=['blue', 'green', 'red'], alpha=0.7)
        ax3.set_title('Action Distribution')
        ax3.set_xlabel('Action')
        ax3.set_ylabel('Percentage (%)')
        ax3.grid(True, alpha=0.3)
        
        # Holdings over time
        ax4.plot(results['dates'], results['holdings'], color='purple', linewidth=2)
        ax4.set_title('Bitcoin Holdings Over Time')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Bitcoin Holdings')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()

def run_comprehensive_backtest():
    """Run comprehensive backtest analysis."""
    model_path = os.path.join("full_training_results", "btc_best_model.pth")
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please ensure the model has been trained first by running train_bitcoin_agent.py")
        return
    
    backtester = BitcoinBacktester(model_path)
    
    # Test on different time periods
    test_periods = [90, 180, 365]  # 3 months, 6 months, 1 year
    
    all_results = []
    
    for period in test_periods:
        try:
            test_data = backtester.get_test_data(days_back=period)
            
            # Test both strategies
            for strategy in ['stochastic', 'deterministic']:
                results = backtester.run_backtest(test_data, strategy=strategy)
                results['test_period_days'] = period
                all_results.append(results)
                
                # Save detailed results
                output_file = f"btc_backtest_{strategy}_{period}d.csv"
                backtester.save_results(results, output_file)
                
                # Generate plots
                plot_file = f"btc_backtest_{strategy}_{period}d.png"
                backtester.plot_results(results, save_path=plot_file)
                
                print(f"\n{strategy.upper()} STRATEGY - {period} DAYS")
                print("="*50)
                print(f"Test Period: {results['test_period']}")
                print(f"Total Return: {results['total_return']:.2f}%")
                print(f"Buy & Hold Return: {results['buy_hold_return']:.2f}%")
                print(f"Outperformance: {results['outperformance']:.2f}%")
                print(f"Final Portfolio Value: ${results['final_value']:,.2f}")
                
                action_analysis = backtester.analyze_actions(results['actions_taken'])
                print(f"Actions: Hold {action_analysis['action_percentages'][0]:.1f}%, "
                      f"Buy {action_analysis['action_percentages'][1]:.1f}%, "
                      f"Sell {action_analysis['action_percentages'][2]:.1f}%")
                
        except Exception as e:
            logger.error(f"Error testing {period} days: {str(e)}")
    
    # Save summary results
    summary_data = []
    for result in all_results:
        summary_data.append({
            'strategy': result['strategy'],
            'test_period_days': result['test_period_days'],
            'total_return': result['total_return'],
            'buy_hold_return': result['buy_hold_return'],
            'outperformance': result['outperformance'],
            'final_value': result['final_value']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('btc_backtest_summary.csv', index=False)
    
    print("\n" + "="*60)
    print("BITCOIN BACKTEST COMPLETED")
    print("="*60)
    print("Summary results saved to btc_backtest_summary.csv")
    print("Detailed results and plots saved for each test period")
    
    return all_results

if __name__ == "__main__":
    try:
        print("Starting Bitcoin DRL Agent Backtest...")
        print("This will test the trained model on multiple time periods")
        print("="*60)
        
        results = run_comprehensive_backtest()
        
    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}")
        raise