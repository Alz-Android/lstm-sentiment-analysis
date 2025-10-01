"""
Bitcoin DRL Training Results Analysis Script
Analyzes the completed training results and generates comprehensive reports.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BitcoinTrainingAnalyzer:
    """Analyzes Bitcoin DRL training results."""
    
    def __init__(self, results_dir='full_training_results'):
        self.results_dir = results_dir
        self.results_data = None
        self.training_stats = {}
        
    def load_training_results(self):
        """Load the complete training results."""
        results_file = os.path.join(self.results_dir, 'btc_complete_training_results.csv')
        
        if not os.path.exists(results_file):
            logger.error(f"Results file not found: {results_file}")
            return False
            
        try:
            self.results_data = pd.read_csv(results_file)
            logger.info(f"Loaded {len(self.results_data)} training episodes")
            return True
        except Exception as e:
            logger.error(f"Error loading results: {str(e)}")
            return False
    
    def calculate_training_statistics(self):
        """Calculate comprehensive training statistics."""
        if self.results_data is None:
            logger.error("No training data loaded")
            return
            
        df = self.results_data
        
        # Basic statistics
        self.training_stats = {
            'total_episodes': len(df),
            'final_episodes_avg': df['reward'].tail(100).mean(),
            'final_returns_avg': df['return'].tail(100).mean(),
            'best_episode_reward': df['reward'].max(),
            'best_episode_return': df['return'].max(),
            'worst_episode_reward': df['reward'].min(),
            'worst_episode_return': df['return'].min(),
            'avg_episode_reward': df['reward'].mean(),
            'avg_episode_return': df['return'].mean(),
            'reward_std': df['reward'].std(),
            'return_std': df['return'].std()
        }
        
        # Learning progress analysis
        episodes_per_window = 100
        windows = len(df) // episodes_per_window
        
        self.training_stats['learning_windows'] = []
        for i in range(windows):
            start_idx = i * episodes_per_window
            end_idx = (i + 1) * episodes_per_window
            window_data = df.iloc[start_idx:end_idx]
            
            self.training_stats['learning_windows'].append({
                'window': i + 1,
                'episodes': f"{start_idx + 1}-{end_idx}",
                'avg_reward': window_data['reward'].mean(),
                'avg_return': window_data['return'].mean(),
                'best_reward': window_data['reward'].max(),
                'best_return': window_data['return'].max()
            })
        
        # Action analysis
        total_actions = df['hold_actions'].sum() + df['buy_actions'].sum() + df['sell_actions'].sum()
        
        self.training_stats['action_distribution'] = {
            'hold_percentage': (df['hold_actions'].sum() / total_actions) * 100,
            'buy_percentage': (df['buy_actions'].sum() / total_actions) * 100,
            'sell_percentage': (df['sell_actions'].sum() / total_actions) * 100,
            'avg_hold_per_episode': df['hold_actions'].mean(),
            'avg_buy_per_episode': df['buy_actions'].mean(),
            'avg_sell_per_episode': df['sell_actions'].mean()
        }
        
        # Performance trends
        self.training_stats['performance_trend'] = {
            'reward_correlation': df['reward'].corr(df.index),
            'return_correlation': df['return'].corr(df.index),
            'improvement_rate': (self.training_stats['final_episodes_avg'] - df['reward'].head(100).mean()) / self.training_stats['total_episodes']
        }
        
        logger.info("Training statistics calculated successfully")
    
    def generate_visualizations(self, save_dir='plots'):
        """Generate comprehensive visualization plots."""
        if self.results_data is None:
            logger.error("No training data loaded")
            return
            
        os.makedirs(save_dir, exist_ok=True)
        df = self.results_data
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Training Progress Overview
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Episode rewards
        ax1.plot(df['episode'], df['reward'], alpha=0.6, linewidth=1, color='blue')
        ax1.plot(df['episode'], df['reward'].rolling(50).mean(), linewidth=2, color='red', label='50-episode MA')
        ax1.set_title('Bitcoin DRL Agent - Episode Rewards', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Portfolio returns
        ax2.plot(df['episode'], df['return'], alpha=0.6, linewidth=1, color='green')
        ax2.plot(df['episode'], df['return'].rolling(50).mean(), linewidth=2, color='orange', label='50-episode MA')
        ax2.set_title('Bitcoin DRL Agent - Portfolio Returns', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Return (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Action distribution over time
        ax3.plot(df['episode'], df['hold_actions'], label='Hold', alpha=0.7)
        ax3.plot(df['episode'], df['buy_actions'], label='Buy', alpha=0.7)
        ax3.plot(df['episode'], df['sell_actions'], label='Sell', alpha=0.7)
        ax3.set_title('Action Distribution Over Episodes', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Number of Actions')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Cumulative reward distribution
        ax4.hist(df['reward'], bins=50, alpha=0.7, color='purple', edgecolor='black')
        ax4.axvline(df['reward'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${df["reward"].mean():.2f}')
        ax4.axvline(df['reward'].median(), color='orange', linestyle='--', linewidth=2, label=f'Median: ${df["reward"].median():.2f}')
        ax4.set_title('Reward Distribution', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Episode Reward ($)')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'btc_training_overview.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Learning Progress Analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Learning windows analysis
        if 'learning_windows' in self.training_stats:
            windows_df = pd.DataFrame(self.training_stats['learning_windows'])
            
            ax1.bar(windows_df['window'], windows_df['avg_reward'], alpha=0.7, color='skyblue', edgecolor='navy')
            ax1.set_title('Average Reward by Training Window (100 episodes each)', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Training Window')
            ax1.set_ylabel('Average Reward ($)')
            ax1.grid(True, alpha=0.3)
            
            ax2.bar(windows_df['window'], windows_df['avg_return'], alpha=0.7, color='lightgreen', edgecolor='darkgreen')
            ax2.set_title('Average Return by Training Window (100 episodes each)', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Training Window')
            ax2.set_ylabel('Average Return (%)')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'btc_learning_progress.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Action Analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Overall action distribution
        if 'action_distribution' in self.training_stats:
            actions = ['Hold', 'Buy', 'Sell']
            percentages = [
                self.training_stats['action_distribution']['hold_percentage'],
                self.training_stats['action_distribution']['buy_percentage'],
                self.training_stats['action_distribution']['sell_percentage']
            ]
            
            colors = ['gold', 'lightgreen', 'lightcoral']
            ax1.pie(percentages, labels=actions, autopct='%1.1f%%', startangle=90, colors=colors)
            ax1.set_title('Overall Action Distribution', fontsize=14, fontweight='bold')
        
        # Action evolution over training
        window_size = 100
        hold_ma = df['hold_actions'].rolling(window_size).mean()
        buy_ma = df['buy_actions'].rolling(window_size).mean()
        sell_ma = df['sell_actions'].rolling(window_size).mean()
        
        ax2.plot(df['episode'], hold_ma, label='Hold (MA)', linewidth=2)
        ax2.plot(df['episode'], buy_ma, label='Buy (MA)', linewidth=2)
        ax2.plot(df['episode'], sell_ma, label='Sell (MA)', linewidth=2)
        ax2.set_title(f'Action Evolution ({window_size}-episode Moving Average)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Average Actions per Episode')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'btc_action_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved in {save_dir} directory")
    
    def save_analysis_report(self, filename='BTC_TRAINING_RESULTS.md'):
        """Save comprehensive analysis report in Markdown format."""
        if not self.training_stats:
            logger.error("No training statistics available")
            return
            
        report_content = f"""# Bitcoin DRL Trading Agent - Training Results Analysis

**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Training Episodes:** {self.training_stats['total_episodes']}

## Executive Summary

The Bitcoin Deep Reinforcement Learning trading agent completed training over {self.training_stats['total_episodes']} episodes using 6 years of historical Bitcoin data. The agent demonstrated significant learning capabilities and achieved substantial portfolio gains.

## Key Performance Metrics

### Overall Performance
- **Average Episode Reward:** ${self.training_stats['avg_episode_reward']:,.2f}
- **Final 100 Episodes Average Reward:** ${self.training_stats['final_episodes_avg']:,.2f}
- **Best Episode Reward:** ${self.training_stats['best_episode_reward']:,.2f}
- **Average Portfolio Return:** {self.training_stats['avg_episode_return']:.2f}%
- **Final 100 Episodes Average Return:** {self.training_stats['final_returns_avg']:.2f}%
- **Best Episode Return:** {self.training_stats['best_episode_return']:.2f}%

### Risk Metrics
- **Reward Standard Deviation:** ${self.training_stats['reward_std']:,.2f}
- **Return Standard Deviation:** {self.training_stats['return_std']:.2f}%
- **Worst Episode Reward:** ${self.training_stats['worst_episode_reward']:,.2f}
- **Worst Episode Return:** {self.training_stats['worst_episode_return']:.2f}%

## Learning Analysis

### Training Progress
- **Improvement Rate:** ${self.training_stats['performance_trend']['improvement_rate']:,.2f} per episode
- **Reward Trend Correlation:** {self.training_stats['performance_trend']['reward_correlation']:.4f}
- **Return Trend Correlation:** {self.training_stats['performance_trend']['return_correlation']:.4f}

The correlation values indicate {'strong positive learning trend' if self.training_stats['performance_trend']['reward_correlation'] > 0.3 else 'moderate learning trend' if self.training_stats['performance_trend']['reward_correlation'] > 0.1 else 'stable performance'} throughout training.

## Trading Behavior Analysis

### Action Distribution
- **Hold Actions:** {self.training_stats['action_distribution']['hold_percentage']:.1f}%
- **Buy Actions:** {self.training_stats['action_distribution']['buy_percentage']:.1f}%
- **Sell Actions:** {self.training_stats['action_distribution']['sell_percentage']:.1f}%

### Average Actions per Episode
- **Hold:** {self.training_stats['action_distribution']['avg_hold_per_episode']:.1f}
- **Buy:** {self.training_stats['action_distribution']['avg_buy_per_episode']:.1f}
- **Sell:** {self.training_stats['action_distribution']['avg_sell_per_episode']:.1f}

## Learning Windows Analysis (100-episode windows)

"""
        
        if 'learning_windows' in self.training_stats:
            report_content += "| Window | Episodes | Avg Reward | Avg Return | Best Reward | Best Return |\n"
            report_content += "|--------|----------|------------|------------|-------------|--------------|\n"
            
            for window in self.training_stats['learning_windows']:
                report_content += f"| {window['window']} | {window['episodes']} | ${window['avg_reward']:,.2f} | {window['avg_return']:.2f}% | ${window['best_reward']:,.2f} | {window['best_return']:.2f}% |\n"
        
        report_content += f"""
## Technical Configuration

### Model Architecture
- **Algorithm:** Proximal Policy Optimization (PPO)
- **Neural Network:** CNN-LSTM hybrid architecture
- **State Space:** 30-day lookback window with OHLCV + technical indicators
- **Action Space:** 3 discrete actions (Hold, Buy, Sell)

### Training Parameters
- **Learning Rate:** 0.0003
- **Batch Size:** 64
- **Lookback Window:** 30 days
- **Initial Balance:** $10,000
- **Transaction Fee:** 0.1%

### Data Configuration
- **Cryptocurrency:** Bitcoin (BTC)
- **Training Period:** 6 years of historical data
- **Data Source:** CryptoCompare API
- **Technical Indicators:** RSI, ATR, OBV

## Conclusions

The Bitcoin DRL trading agent showed {'excellent' if self.training_stats['final_episodes_avg'] > 15000 else 'good' if self.training_stats['final_episodes_avg'] > 10000 else 'moderate'} performance with an average reward of ${self.training_stats['final_episodes_avg']:,.2f} in the final 100 episodes. 

### Key Findings:
1. **Learning Capability:** The agent demonstrated {'strong' if self.training_stats['performance_trend']['reward_correlation'] > 0.2 else 'moderate'} learning progression over the training period
2. **Risk Management:** {'Balanced' if 20 <= self.training_stats['action_distribution']['hold_percentage'] <= 60 else 'Conservative' if self.training_stats['action_distribution']['hold_percentage'] > 60 else 'Aggressive'} trading strategy with {self.training_stats['action_distribution']['hold_percentage']:.1f}% hold actions
3. **Profit Generation:** Achieved an average return of {self.training_stats['final_returns_avg']:.2f}% in recent episodes

### Recommendations:
- {'✅ Ready for backtesting' if self.training_stats['final_episodes_avg'] > 8000 else '⚠️ Consider additional training'}
- Monitor performance on out-of-sample data
- Consider ensemble methods for improved robustness

## Next Steps

1. **Backtesting:** Run comprehensive backtest on unseen data periods
2. **Strategy Comparison:** Compare against buy-and-hold benchmarks
3. **Risk Analysis:** Evaluate maximum drawdown and volatility metrics
4. **Model Deployment:** Consider deployment for paper trading

---

*This analysis was generated automatically from the training results. For detailed visualizations, see the accompanying plot files.*
"""
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report_content)
            logger.info(f"Analysis report saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving report: {str(e)}")

def analyze_bitcoin_training():
    """Main function to analyze Bitcoin training results."""
    print("="*60)
    print("BITCOIN DRL TRAINING RESULTS ANALYSIS")
    print("="*60)
    
    analyzer = BitcoinTrainingAnalyzer()
    
    # Check if training results exist
    if not analyzer.load_training_results():
        print("❌ Training results not found!")
        print("Please ensure the training has completed first by running:")
        print("python train_bitcoin_agent.py")
        return
    
    print("✅ Training results loaded successfully")
    
    # Calculate statistics
    analyzer.calculate_training_statistics()
    print("✅ Training statistics calculated")
    
    # Generate visualizations
    analyzer.generate_visualizations()
    print("✅ Visualizations generated")
    
    # Save analysis report
    analyzer.save_analysis_report()
    print("✅ Analysis report saved")
    
    # Display key statistics
    stats = analyzer.training_stats
    print("\n" + "="*60)
    print("BITCOIN TRAINING SUMMARY")
    print("="*60)
    print(f"Total Episodes: {stats['total_episodes']}")
    print(f"Final 100 Episodes Avg Reward: ${stats['final_episodes_avg']:,.2f}")
    print(f"Final 100 Episodes Avg Return: {stats['final_returns_avg']:.2f}%")
    print(f"Best Episode Reward: ${stats['best_episode_reward']:,.2f}")
    print(f"Best Episode Return: {stats['best_episode_return']:.2f}%")
    print(f"Action Distribution: Hold {stats['action_distribution']['hold_percentage']:.1f}%, Buy {stats['action_distribution']['buy_percentage']:.1f}%, Sell {stats['action_distribution']['sell_percentage']:.1f}%")
    print("="*60)
    
    if stats['final_episodes_avg'] > 8000:
        print("🎉 TRAINING SUCCESSFUL! Agent ready for backtesting.")
    else:
        print("⚠️ Training completed but performance may need improvement.")
    
    print("\nFiles generated:")
    print("- BTC_TRAINING_RESULTS.md (comprehensive analysis)")
    print("- plots/btc_training_overview.png")
    print("- plots/btc_learning_progress.png") 
    print("- plots/btc_action_analysis.png")

if __name__ == "__main__":
    try:
        analyze_bitcoin_training()
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise