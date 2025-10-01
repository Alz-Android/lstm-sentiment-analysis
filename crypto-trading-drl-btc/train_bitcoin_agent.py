"""
Bitcoin DRL Trading Agent Training Script
Trains a PPO agent to trade Bitcoin using historical data with technical indicators.
"""

import os
import sys
import logging
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_fetcher import DataFetcher
from trading_env import TradingEnvironment
from neural_networks import PPOAgent
import numpy as np
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'btc_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_training_environment():
    """Set up the Bitcoin trading environment with historical data."""
    logger.info("Setting up Bitcoin training environment...")
    
    # Fetch Bitcoin historical data
    data_fetcher = DataFetcher()
    logger.info("Fetching Bitcoin historical data (6 years)...")
    
    # Fetch 6 years of daily data for comprehensive training
    btc_data = data_fetcher.fetch_daily_data(
        symbol='BTC',
        days=2190  # ~6 years
    )
    
    if btc_data is None or len(btc_data) < 100:
        raise ValueError("Insufficient Bitcoin data for training")
    
    logger.info(f"Fetched {len(btc_data)} days of Bitcoin data")
    logger.info(f"Date range: {btc_data.index[0]} to {btc_data.index[-1]}")
    logger.info(f"Price range: ${btc_data['close'].min():.2f} - ${btc_data['close'].max():.2f}")
    
    # Save data for reference
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    btc_data.to_csv(os.path.join(data_dir, 'btc_training_data.csv'))
    logger.info(f"Saved training data to {os.path.join(data_dir, 'btc_training_data.csv')}")
    
    # Add technical indicators
    from technical_indicators import TechnicalIndicators
    ti = TechnicalIndicators()
    btc_data = ti.add_technical_indicators(btc_data)
    logger.info("Added technical indicators to Bitcoin data")
    
    # Create trading environment
    env = TradingEnvironment(
        df=btc_data,
        initial_balance=10000,
        lookback_window=30
    )
    
    logger.info("Bitcoin trading environment created successfully")
    return env, btc_data

def train_bitcoin_agent(episodes=1000, save_interval=100):
    """Train the Bitcoin DRL agent."""
    logger.info(f"Starting Bitcoin agent training for {episodes} episodes...")
    
    # Setup environment
    env, data = setup_training_environment()
    
    # Create PPO agent
    state_shape = env.observation_space.shape  # (30, 4)
    action_size = env.action_space.n
    
    agent = PPOAgent(
        input_shape=state_shape,
        action_dim=action_size,
        lr=0.0003
    )
    
    logger.info(f"Created PPO agent with state_shape={state_shape}, action_size={action_size}")
    
    # Training metrics
    episode_rewards = []
    episode_returns = []
    episode_actions = []
    best_reward = float('-inf')
    
    # Create results directory
    results_dir = "full_training_results"
    os.makedirs(results_dir, exist_ok=True)
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_action_counts = [0, 0, 0]  # [Hold, Buy, Sell]
        done = False
        
        while not done:
            action, log_prob = agent.act(state)
            next_state, reward, done, info = env.step(action)
            
            agent.store_transition(state, action, log_prob, reward, done)
            
            state = next_state
            episode_reward += reward
            episode_action_counts[action] += 1
        
        # Calculate portfolio return
        portfolio_return = ((env.current_balance + env.current_holdings * env.current_price) / env.initial_balance - 1) * 100
        
        episode_rewards.append(episode_reward)
        episode_returns.append(portfolio_return)
        episode_actions.append(episode_action_counts)
        
        # Update agent
        if len(agent.memory) > agent.batch_size:
            agent.update()
        
        # Logging
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_return = np.mean(episode_returns[-10:])
            logger.info(f"Episode {episode:4d} | Reward: {episode_reward:8.2f} | Return: {portfolio_return:6.2f}% | Avg Reward: {avg_reward:8.2f} | Avg Return: {avg_return:6.2f}%")
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save_model(os.path.join(results_dir, 'btc_best_model.pth'))
            logger.info(f"New best model saved with reward: {best_reward:.2f}")
        
        # Save checkpoint
        if episode % save_interval == 0 and episode > 0:
            agent.save_model(os.path.join(results_dir, f'btc_model_episode_{episode}.pth'))
            
            # Save training progress
            progress_data = {
                'episode': list(range(len(episode_rewards))),
                'reward': episode_rewards,
                'return': episode_returns,
                'hold_actions': [actions[0] for actions in episode_actions],
                'buy_actions': [actions[1] for actions in episode_actions],
                'sell_actions': [actions[2] for actions in episode_actions]
            }
            
            progress_df = pd.DataFrame(progress_data)
            progress_df.to_csv(os.path.join(results_dir, f'btc_training_progress_episode_{episode}.csv'), index=False)
            
            logger.info(f"Checkpoint saved at episode {episode}")
    
    # Save final model and results
    agent.save_model(os.path.join(results_dir, 'btc_final_model.pth'))
    
    # Save complete training results
    final_results = {
        'episode': list(range(len(episode_rewards))),
        'reward': episode_rewards,
        'return': episode_returns,
        'hold_actions': [actions[0] for actions in episode_actions],
        'buy_actions': [actions[1] for actions in episode_actions],
        'sell_actions': [actions[2] for actions in episode_actions]
    }
    
    results_df = pd.DataFrame(final_results)
    results_df.to_csv(os.path.join(results_dir, 'btc_complete_training_results.csv'), index=False)
    
    # Training summary
    final_avg_reward = np.mean(episode_rewards[-100:])  # Last 100 episodes
    final_avg_return = np.mean(episode_returns[-100:])
    max_reward = max(episode_rewards)
    max_return = max(episode_returns)
    
    logger.info("="*80)
    logger.info("BITCOIN TRAINING COMPLETED")
    logger.info("="*80)
    logger.info(f"Total Episodes: {episodes}")
    logger.info(f"Final Average Reward (last 100): ${final_avg_reward:,.2f}")
    logger.info(f"Final Average Return (last 100): {final_avg_return:.2f}%")
    logger.info(f"Best Episode Reward: ${max_reward:,.2f}")
    logger.info(f"Best Episode Return: {max_return:.2f}%")
    logger.info(f"Models saved in: {results_dir}")
    logger.info("="*80)
    
    return agent, results_df, env

if __name__ == "__main__":
    try:
        print("Starting Bitcoin DRL Trading Agent Training...")
        print("This will train for 1000 episodes with comprehensive logging.")
        print("Training progress will be saved every 100 episodes.")
        print("="*60)
        
        # Run training
        agent, results, env = train_bitcoin_agent(episodes=1000, save_interval=100)
        
        print("\nTraining completed successfully!")
        print("Check 'full_training_results' directory for saved models and results.")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise