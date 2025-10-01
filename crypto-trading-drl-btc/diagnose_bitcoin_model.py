"""
Bitcoin DRL Model Diagnostic Script
Analyzes the trained Bitcoin model's behavior and action distribution.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_fetcher import DataFetcher
from trading_env import TradingEnvironment
from neural_networks import PPOAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BitcoinModelDiagnostic:
    """Diagnostic tool for analyzing Bitcoin DRL model behavior."""
    
    def __init__(self, model_path=None):
        self.model_path = model_path or os.path.join("full_training_results", "btc_best_model.pth")
        self.agent = None
        self.test_env = None
        
    def load_model(self):
        """Load the trained Bitcoin model."""
        if not os.path.exists(self.model_path):
            logger.error(f"Model file not found: {self.model_path}")
            return False
            
        try:
            # Create environment to get state and action dimensions
            data_fetcher = DataFetcher()
            test_data = data_fetcher.fetch_daily_data('BTC', days=90)
            
            if test_data is None or len(test_data) < 60:
                logger.error("Insufficient test data for model diagnostics")
                return False
                
            self.test_env = TradingEnvironment(
                df=test_data,
                initial_balance=10000,
                lookback_window=30
            )
            
            # Initialize agent with correct dimensions
            state_shape = self.test_env.observation_space.shape
            action_size = self.test_env.action_space.n
            
            self.agent = PPOAgent(
                input_shape=state_shape,
                action_dim=action_size,
                lr=0.0003
            )
            
            # Load the trained model
            import torch
            checkpoint = torch.load(self.model_path, map_location='cpu')
            if 'actor_state_dict' in checkpoint:
                self.agent.actor.load_state_dict(checkpoint['actor_state_dict'])
                self.agent.critic.load_state_dict(checkpoint['critic_state_dict'])
            else:
                # Assume the checkpoint contains just the model state dict
                self.agent.actor.load_state_dict(checkpoint)
                
            logger.info("Bitcoin model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def analyze_action_distribution(self, num_samples=1000):
        """Analyze the model's action distribution on random states."""
        if self.agent is None or self.test_env is None:
            logger.error("Model not loaded")
            return None
            
        action_counts = [0, 0, 0]  # [Hold, Buy, Sell]
        action_probs_list = []
        
        # Reset environment and collect samples
        state = self.test_env.reset()
        
        for i in range(min(num_samples, len(self.test_env.df) - self.test_env.lookback_window - 1)):
            # Get action probabilities
            action, log_prob, action_probs = self.agent.get_action(state, deterministic=False)
            action_counts[action] += 1
            action_probs_list.append(action_probs)
            
            # Step environment
            state, _, done, _ = self.test_env.step(action)
            if done:
                state = self.test_env.reset()
        
        # Calculate statistics
        total_samples = sum(action_counts)
        action_percentages = [count/total_samples*100 for count in action_counts]
        
        # Average action probabilities
        avg_action_probs = np.mean(action_probs_list, axis=0) * 100
        
        results = {
            'action_counts': action_counts,
            'action_percentages': action_percentages,
            'avg_action_probabilities': avg_action_probs,
            'total_samples': total_samples
        }
        
        logger.info(f"Action distribution analysis completed over {total_samples} samples")
        return results
    
    def test_deterministic_vs_stochastic(self, num_episodes=10):
        """Compare deterministic vs stochastic action selection."""
        if self.agent is None or self.test_env is None:
            logger.error("Model not loaded")
            return None
            
        results = {
            'deterministic': {'episodes': [], 'actions': []},
            'stochastic': {'episodes': [], 'actions': []}
        }
        
        for mode in ['deterministic', 'stochastic']:
            for episode in range(num_episodes):
                state = self.test_env.reset()
                episode_actions = []
                done = False
                episode_reward = 0
                
                while not done:
                    deterministic = (mode == 'deterministic')
                    action, _, _ = self.agent.get_action(state, deterministic=deterministic)
                    episode_actions.append(action)
                    
                    state, reward, done, info = self.test_env.step(action)
                    episode_reward += reward
                
                # Count actions
                action_counts = [episode_actions.count(i) for i in range(3)]
                results[mode]['episodes'].append({
                    'episode': episode,
                    'reward': episode_reward,
                    'final_value': info['net_worth'],
                    'return': (info['net_worth'] / 10000 - 1) * 100,
                    'action_counts': action_counts,
                    'total_actions': len(episode_actions)
                })
                results[mode]['actions'].extend(episode_actions)
        
        # Calculate overall statistics
        for mode in results:
            total_actions = len(results[mode]['actions'])
            action_counts = [results[mode]['actions'].count(i) for i in range(3)]
            action_percentages = [count/total_actions*100 for count in action_counts]
            
            avg_reward = np.mean([ep['reward'] for ep in results[mode]['episodes']])
            avg_return = np.mean([ep['return'] for ep in results[mode]['episodes']])
            
            results[mode]['summary'] = {
                'total_actions': total_actions,
                'action_counts': action_counts,
                'action_percentages': action_percentages,
                'avg_reward': avg_reward,
                'avg_return': avg_return
            }
        
        logger.info("Deterministic vs Stochastic comparison completed")
        return results
    
    def analyze_state_value_predictions(self, num_samples=100):
        """Analyze the critic's state value predictions."""
        if self.agent is None or self.test_env is None:
            logger.error("Model not loaded")
            return None
            
        import torch
        
        state_values = []
        actual_rewards = []
        
        state = self.test_env.reset()
        
        for i in range(min(num_samples, len(self.test_env.df) - self.test_env.lookback_window - 1)):
            # Get state value from critic
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                value = self.agent.critic(state_tensor).item()
            state_values.append(value)
            
            # Take action and get actual reward
            action, _, _ = self.agent.get_action(state, deterministic=False)
            state, reward, done, _ = self.test_env.step(action)
            actual_rewards.append(reward)
            
            if done:
                state = self.test_env.reset()
        
        results = {
            'state_values': state_values,
            'actual_rewards': actual_rewards,
            'value_stats': {
                'mean': np.mean(state_values),
                'std': np.std(state_values),
                'min': np.min(state_values),
                'max': np.max(state_values)
            },
            'reward_stats': {
                'mean': np.mean(actual_rewards),
                'std': np.std(actual_rewards),
                'min': np.min(actual_rewards),
                'max': np.max(actual_rewards)
            },
            'correlation': np.corrcoef(state_values, actual_rewards)[0, 1]
        }
        
        logger.info("State value analysis completed")
        return results
    
    def generate_diagnostic_report(self, save_path="BTC_MODEL_DIAGNOSTIC.md"):
        """Generate comprehensive diagnostic report."""
        
        # Run all analyses
        action_dist = self.analyze_action_distribution()
        det_vs_stoch = self.test_deterministic_vs_stochastic()
        value_analysis = self.analyze_state_value_predictions()
        
        if not all([action_dist, det_vs_stoch, value_analysis]):
            logger.error("Some analyses failed")
            return False
        
        # Generate report
        report = f"""# Bitcoin DRL Model Diagnostic Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Model:** {self.model_path}

## Executive Summary

This diagnostic report analyzes the trained Bitcoin DRL model's behavior, action selection patterns, and value predictions.

## Action Distribution Analysis

### Overall Action Preferences (1000 samples)
- **Hold:** {action_dist['action_percentages'][0]:.1f}% ({action_dist['action_counts'][0]} actions)
- **Buy:** {action_dist['action_percentages'][1]:.1f}% ({action_dist['action_counts'][1]} actions)  
- **Sell:** {action_dist['action_percentages'][2]:.1f}% ({action_dist['action_counts'][2]} actions)

### Average Action Probabilities
- **Hold:** {action_dist['avg_action_probabilities'][0]:.1f}%
- **Buy:** {action_dist['avg_action_probabilities'][1]:.1f}%
- **Sell:** {action_dist['avg_action_probabilities'][2]:.1f}%

## Deterministic vs Stochastic Comparison

### Deterministic Mode
- **Average Reward:** ${det_vs_stoch['deterministic']['summary']['avg_reward']:.2f}
- **Average Return:** {det_vs_stoch['deterministic']['summary']['avg_return']:.2f}%
- **Action Distribution:**
  - Hold: {det_vs_stoch['deterministic']['summary']['action_percentages'][0]:.1f}%
  - Buy: {det_vs_stoch['deterministic']['summary']['action_percentages'][1]:.1f}%
  - Sell: {det_vs_stoch['deterministic']['summary']['action_percentages'][2]:.1f}%

### Stochastic Mode
- **Average Reward:** ${det_vs_stoch['stochastic']['summary']['avg_reward']:.2f}
- **Average Return:** {det_vs_stoch['stochastic']['summary']['avg_return']:.2f}%
- **Action Distribution:**
  - Hold: {det_vs_stoch['stochastic']['summary']['action_percentages'][0]:.1f}%
  - Buy: {det_vs_stoch['stochastic']['summary']['action_percentages'][1]:.1f}%
  - Sell: {det_vs_stoch['stochastic']['summary']['action_percentages'][2]:.1f}%

## Value Function Analysis

### State Value Predictions
- **Mean Value:** {value_analysis['value_stats']['mean']:.2f}
- **Standard Deviation:** {value_analysis['value_stats']['std']:.2f}
- **Range:** [{value_analysis['value_stats']['min']:.2f}, {value_analysis['value_stats']['max']:.2f}]

### Actual Rewards
- **Mean Reward:** {value_analysis['reward_stats']['mean']:.2f}
- **Standard Deviation:** {value_analysis['reward_stats']['std']:.2f}
- **Range:** [{value_analysis['reward_stats']['min']:.2f}, {value_analysis['reward_stats']['max']:.2f}]

### Value-Reward Correlation
- **Correlation Coefficient:** {value_analysis['correlation']:.4f}

## Model Behavior Assessment

### Strategy Classification
"""

        # Assess strategy type based on action distribution
        hold_pct = action_dist['action_percentages'][0]
        buy_pct = action_dist['action_percentages'][1]
        sell_pct = action_dist['action_percentages'][2]
        
        if hold_pct > 60:
            strategy_type = "Conservative (Hold-Heavy)"
        elif buy_pct > sell_pct and buy_pct > 30:
            strategy_type = "Bullish (Buy-Oriented)"
        elif sell_pct > buy_pct and sell_pct > 30:
            strategy_type = "Bearish (Sell-Oriented)"
        else:
            strategy_type = "Balanced"
            
        report += f"**Strategy Type:** {strategy_type}\n\n"
        
        # Performance assessment
        stoch_return = det_vs_stoch['stochastic']['summary']['avg_return']
        det_return = det_vs_stoch['deterministic']['summary']['avg_return']
        
        report += f"""### Performance Assessment
- **Preferred Mode:** {'Stochastic' if stoch_return > det_return else 'Deterministic'} (Higher returns)
- **Return Difference:** {abs(stoch_return - det_return):.2f}%
- **Value Function Quality:** {'Good' if abs(value_analysis['correlation']) > 0.3 else 'Moderate' if abs(value_analysis['correlation']) > 0.1 else 'Poor'} (Correlation: {value_analysis['correlation']:.3f})

## Recommendations

"""
        
        # Generate recommendations
        if stoch_return > 0 and det_return > 0:
            report += "✅ **Model Performance:** Both modes show positive returns\n"
        elif stoch_return > 0:
            report += "⚠️ **Model Performance:** Only stochastic mode shows positive returns\n"
        else:
            report += "❌ **Model Performance:** Both modes show negative returns - model needs improvement\n"
            
        if abs(stoch_return - det_return) > 5:
            report += "⚠️ **Mode Sensitivity:** Large difference between deterministic and stochastic modes\n"
        else:
            report += "✅ **Mode Consistency:** Similar performance between modes\n"
            
        if abs(value_analysis['correlation']) > 0.3:
            report += "✅ **Value Function:** Strong correlation with actual rewards\n"
        else:
            report += "⚠️ **Value Function:** Weak correlation with actual rewards - may need more training\n"
            
        report += f"""
## Technical Details

- **Model Architecture:** CNN-LSTM with PPO
- **State Space:** {self.test_env.observation_space.shape}
- **Action Space:** {self.test_env.action_space.n} discrete actions
- **Analysis Date:** {datetime.now().strftime('%Y-%m-%d')}

---
*This diagnostic report was generated automatically from model analysis.*
"""
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Diagnostic report saved to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving report: {str(e)}")
            return False

def diagnose_bitcoin_model():
    """Main function to diagnose Bitcoin model."""
    print("="*60)
    print("BITCOIN DRL MODEL DIAGNOSTIC")
    print("="*60)
    
    diagnostic = BitcoinModelDiagnostic()
    
    if not diagnostic.load_model():
        print("❌ Failed to load Bitcoin model")
        print("Please ensure training has completed and model file exists")
        return
        
    print("✅ Bitcoin model loaded successfully")
    
    # Run diagnostic analyses
    print("🔍 Running action distribution analysis...")
    action_dist = diagnostic.analyze_action_distribution()
    
    print("🔍 Running deterministic vs stochastic comparison...")
    det_vs_stoch = diagnostic.test_deterministic_vs_stochastic()
    
    print("🔍 Running value function analysis...")
    value_analysis = diagnostic.analyze_state_value_predictions()
    
    if all([action_dist, det_vs_stoch, value_analysis]):
        print("✅ All analyses completed successfully")
        
        # Generate and save report
        if diagnostic.generate_diagnostic_report():
            print("✅ Diagnostic report generated: BTC_MODEL_DIAGNOSTIC.md")
        else:
            print("❌ Failed to generate diagnostic report")
            
        # Display key findings
        print("\n" + "="*60)
        print("KEY FINDINGS")
        print("="*60)
        print(f"Action Distribution: Hold {action_dist['action_percentages'][0]:.1f}%, Buy {action_dist['action_percentages'][1]:.1f}%, Sell {action_dist['action_percentages'][2]:.1f}%")
        print(f"Stochastic Return: {det_vs_stoch['stochastic']['summary']['avg_return']:.2f}%")
        print(f"Deterministic Return: {det_vs_stoch['deterministic']['summary']['avg_return']:.2f}%")
        print(f"Value-Reward Correlation: {value_analysis['correlation']:.3f}")
        
        # Overall assessment
        stoch_return = det_vs_stoch['stochastic']['summary']['avg_return']
        if stoch_return > 2:
            print("🎉 Model shows good performance!")
        elif stoch_return > 0:
            print("✅ Model shows positive returns")
        else:
            print("⚠️ Model needs improvement - negative returns")
            
    else:
        print("❌ Some analyses failed")

if __name__ == "__main__":
    try:
        diagnose_bitcoin_model()
    except Exception as e:
        logger.error(f"Diagnostic failed: {str(e)}")
        raise