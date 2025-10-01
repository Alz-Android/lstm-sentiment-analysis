# Bitcoin Deep Reinforcement Learning Trading Bot

A sophisticated Deep Reinforcement Learning (DRL) trading system specifically designed for Bitcoin trading using Proximal Policy Optimization (PPO) with CNN-LSTM neural networks.

## Features

- **Advanced DRL Agent**: PPO (Proximal Policy Optimization) algorithm with CNN-LSTM architecture
- **Technical Indicators**: RSI, ATR, OBV integrated for enhanced market analysis
- **Comprehensive Backtesting**: Multiple time period testing with both stochastic and deterministic strategies
- **Risk Management**: Transaction fees, position sizing, and portfolio management
- **Real-time Data**: Integration with CryptoCompare API for live Bitcoin data
- **Extensive Logging**: Detailed training progress and performance tracking

## System Architecture

### Core Components

1. **Trading Environment** (`src/trading_env.py`)
   - Custom OpenAI Gym environment for Bitcoin trading
   - Technical indicators integration
   - Portfolio management and risk controls

2. **Neural Networks** (`src/neural_networks.py`)
   - CNN-LSTM architecture for price pattern recognition
   - PPO agent implementation
   - Advanced memory management

3. **Data Management** (`src/data_fetcher.py`)
   - CryptoCompare API integration
   - Historical data preprocessing
   - Technical indicators calculation

4. **Technical Analysis** (`src/technical_indicators.py`)
   - RSI (Relative Strength Index)
   - ATR (Average True Range)
   - OBV (On-Balance Volume)

## Quick Start

### Installation

```bash
pip install -r BTC_requirements.txt
```

### Training the Agent

```bash
python train_bitcoin_agent.py
```

This will train the agent for 1000 episodes using 6 years of historical Bitcoin data.

### Running Backtest

```bash
python backtest_bitcoin_agent.py
```

This will test the trained model on multiple time periods (3 months, 6 months, 1 year).

### Testing the System

```bash
python BTC_test_system.py
```

## Configuration

### Training Parameters

- **Episodes**: 1000 (full training)
- **Learning Rate**: 0.0003
- **Batch Size**: 64
- **Lookback Window**: 30 days
- **Initial Balance**: $10,000
- **Transaction Fee**: 0.1%

### Data Configuration

- **Symbol**: BTC (Bitcoin)
- **Data Source**: CryptoCompare API
- **Training Period**: 6 years of historical data
- **Data Frequency**: Daily OHLCV data

## Results Structure

```
crypto-trading-drl-btc/
├── full_training_results/
│   ├── btc_best_model.pth
│   ├── btc_final_model.pth
│   ├── btc_complete_training_results.csv
│   └── btc_training_progress_*.csv
├── data/
│   └── btc_training_data.csv
├── backtest results (CSV and PNG files)
└── logs/
```

## Performance Metrics

The system tracks comprehensive metrics:

- **Portfolio Returns**: Percentage gains/losses
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Action Distribution**: Hold/Buy/Sell percentages
- **Benchmark Comparison**: vs Buy & Hold strategy

## Technical Specifications

### State Space
- Price data (OHLCV)
- Technical indicators (RSI, ATR, OBV)
- Portfolio status (cash, holdings, value)
- Market volatility measures

### Action Space
- 0: Hold (maintain current position)
- 1: Buy (purchase Bitcoin with available cash)
- 2: Sell (sell Bitcoin holdings)

### Reward Function
- Portfolio value change
- Risk-adjusted returns
- Transaction cost penalties

## Advanced Features

### Stochastic vs Deterministic Trading
- **Stochastic**: Samples actions from probability distribution
- **Deterministic**: Always selects highest probability action
- Comparison analysis included in backtest results

### Risk Management
- Position sizing based on available capital
- Transaction fee incorporation
- Portfolio rebalancing constraints

## Monitoring and Analysis

### Training Monitoring
- Real-time training progress logging
- Episode reward tracking
- Action distribution analysis
- Model checkpoint saving

### Backtest Analysis
- Multiple time period testing
- Strategy comparison (stochastic vs deterministic)
- Performance visualization
- Detailed CSV exports

## Troubleshooting

### Common Issues

1. **Data Fetching Errors**
   - Check internet connection
   - Verify CryptoCompare API accessibility
   - Ensure sufficient historical data

2. **Training Issues**
   - Monitor GPU/CPU usage
   - Check memory availability
   - Verify model saving permissions

3. **Backtest Failures**
   - Ensure trained model exists
   - Check test data availability
   - Verify file permissions

## License

This project is for educational and research purposes. Use at your own risk for live trading.

## Disclaimer

This software is for educational purposes only. Cryptocurrency trading involves substantial risk of loss. Past performance does not guarantee future results. Always conduct your own research before making investment decisions.