"""
Bitcoin DRL Trading Agent - Reduced Training Script
Trains with reduced parameters for initial validation (following lessons learned).
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
        logging.FileHandler(f'btc_reduced_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_reduced_training_environment():
    """Set up the Bitcoin trading environment with reduced historical data."""
    logger.info("Setting up Bitcoin reduced training environment...")
    
    # Fetch Bitcoin historical data - REDUCED dataset (1 year instead of 6)
    data_fetcher = DataFetcher()
    logger.info("Fetching Bitcoin historical data (1 year for initial validation)...")
    
    # Fetch 1 year of daily data for reduced training
    btc_data = data_fetcher.fetch_daily_data(
        symbol='BTC',
        days=365  # 1 year instead of 6
    )
    
    if btc_data is None or len(btc_data) < 100:
        raise ValueError("Insufficient Bitcoin data for training")
    
    logger.info(f"Fetched {len(btc_data)} days of Bitcoin data")
    logger.info(f"Date range: {btc_data.index[0]} to {btc_data.index[-1]}")
    logger.info(f"Price range: ${btc_data['close'].min():.2f} - ${btc_data['close'].max():.2f}")
    
    # Save data for reference
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    btc_data.to_csv(os.path.join(data_dir, 'btc_reduced_training_data.csv'))
    logger.info(f"Saved training data to {os.path.join(data_dir, 'btc_reduced_training_data.csv')}")
    
    # Add technical indicators
    from technical_indicators import TechnicalIndicators
    ti = TechnicalIndicators()
    btc_data = ti.add_technical_indicators(btc_data)
    logger.info("Added technical indicators to Bitcoin data")
    
    # Create trading environment with REDUCED lookback window
    env = TradingEnvironment(
        df=btc_data,
        initial_balance=10000,
        lookback_window=30  # Reduced from 100 to 30
    )
    
    logger.info("Bitcoin reduced training environment created successfully")
    return env, btc_data

def train_bitcoin_agent_reduced(episodes=100, save_interval=25):
    """Train the Bitcoin DRL agent with reduced parameters."""
    logger.info(f"Starting Bitcoin agent REDUCED training for {episodes} episodes...")
    
    # Setup environment
    env, data = setup_reduced_training_environment()
    
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
    results_dir = "reduced_training_results"
    os.makedirs(results_dir, exist_ok=True)
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_action_counts = [0, 0, 0]  # [Hold, Buy, Sell]
        done = False
        
        while not done:
            try:
                action, log_prob = agent.act(state)
                next_state, reward, done, info = env.step(action)
                
                agent.store_transition(state, action, log_prob, reward, done)
                
                state = next_state
                episode_reward += reward
                episode_action_counts[action] += 1
            except Exception as e:
                logger.error(f"Error in episode {episode}, step: {e}")
                break
        
        # Calculate portfolio return
        portfolio_return = ((env.current_balance + env.current_holdings * env.current_price) / env.initial_balance - 1) * 100
        
        episode_rewards.append(episode_reward)
        episode_returns.append(portfolio_return)
        episode_actions.append(episode_action_counts)
        
        # Update agent
        try:
            if len(agent.memory) > agent.batch_size:
                agent.update()
        except Exception as e:
            logger.warning(f"Update failed in episode {episode}: {e}")
        
        # Logging
        if episode % 5 == 0:
            avg_reward = np.mean(episode_rewards[-5:])
            avg_return = np.mean(episode_returns[-5:])
            logger.info(f"Episode {episode:4d} | Reward: {episode_reward:8.2f} | Return: {portfolio_return:6.2f}% | Avg Reward: {avg_reward:8.2f} | Avg Return: {avg_return:6.2f}%")
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            try:
                agent.save_model(os.path.join(results_dir, 'btc_best_model_reduced.pth'))
                logger.info(f"New best model saved with reward: {best_reward:.2f}")
            except Exception as e:
                logger.warning(f"Model save failed: {e}")
        
        # Save checkpoint
        if episode % save_interval == 0 and episode > 0:
            try:
                agent.save_model(os.path.join(results_dir, f'btc_model_episode_{episode}_reduced.pth'))
                
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
                progress_df.to_csv(os.path.join(results_dir, f'btc_training_progress_episode_{episode}_reduced.csv'), index=False)
                
                logger.info(f"Checkpoint saved at episode {episode}")
            except Exception as e:
                logger.warning(f"Checkpoint save failed: {e}")
    
    # Save final model and results
    try:
        agent.save_model(os.path.join(results_dir, 'btc_final_model_reduced.pth'))
        
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
        results_df.to_csv(os.path.join(results_dir, 'btc_complete_training_results_reduced.csv'), index=False)
        
        # Training summary
        final_avg_reward = np.mean(episode_rewards[-10:])  # Last 10 episodes
        final_avg_return = np.mean(episode_returns[-10:])
        max_reward = max(episode_rewards) if episode_rewards else 0
        max_return = max(episode_returns) if episode_returns else 0
        
        logger.info("="*80)
        logger.info("BITCOIN REDUCED TRAINING COMPLETED")
        logger.info("="*80)
        logger.info(f"Total Episodes: {episodes}")
        logger.info(f"Final Average Reward (last 10): ${final_avg_reward:,.2f}")
        logger.info(f"Final Average Return (last 10): {final_avg_return:.2f}%")
        logger.info(f"Best Episode Reward: ${max_reward:,.2f}")
        logger.info(f"Best Episode Return: {max_return:.2f}%")
        logger.info(f"Models saved in: {results_dir}")
        logger.info("="*80)
        
        return agent, results_df, env
        
    except Exception as e:
        logger.error(f"Final save failed: {e}")
        return agent, None, env

if __name__ == "__main__":
    try:
        print("Starting Bitcoin DRL Trading Agent - REDUCED Training...")
        print("This will train for 100 episodes with 1 year of data for validation.")
        print("Designed to quickly validate the pipeline before full training.")
        print("="*60)
        
        # Run reduced training
        agent, results, env = train_bitcoin_agent_reduced(episodes=100, save_interval=25)
        
        print("\nReduced training completed successfully!")
        print("Check 'reduced_training_results' directory for saved models and results.")
        print("If successful, proceed with full training using train_bitcoin_agent.py")
        
    except Exception as e:
        logger.error(f"Reduced training failed: {str(e)}")
        print(f"Training failed: {str(e)}")
        raise