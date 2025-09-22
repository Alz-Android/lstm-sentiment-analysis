# LSTM Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A complete, production-ready sentiment analysis system using LSTM (Long Short-Term Memory) neural networks with PyTorch. This project provides an end-to-end pipeline for binary sentiment classification with advanced features like bidirectional LSTM, attention mechanisms, and comprehensive evaluation tools.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/Alz-Android/lstm-sentiment-analysis.git
cd lstm-sentiment-analysis

# Install dependencies
pip install -r requirements.txt

# Download required NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"

# Run the demo
python demo.py
```

## 📋 Features

- **Complete ML Pipeline**: From raw text input to trained model and predictions
- **Advanced Architecture**: Bidirectional LSTM with optional attention mechanisms
- **Rich Visualizations**: Training progress, ROC curves, and confusion matrices
- **Production API**: Single and batch prediction capabilities with confidence scoring
- **Model Persistence**: Save and load trained models
- **Comprehensive Evaluation**: Detailed metrics and performance reports

## 🏗️ Architecture

The project follows a modular design with clear separation of concerns:

- `data_preprocessing.py` - Text cleaning, tokenization, and vocabulary building
- `lstm_model.py` - Neural network architecture definitions
- `train_model.py` - Training pipeline with early stopping and scheduling
- `evaluate_model.py` - Model evaluation and inference API
- `demo.py` - Interactive demonstration of the complete pipeline
- `test_implementation.py` - Basic functionality validation

## 📊 Model Architecture

- **Embedding Layer**: Configurable word embeddings
- **Bidirectional LSTM**: Captures context from both directions
- **Attention Mechanism**: Optional attention for improved performance
- **Dropout Regularization**: Prevents overfitting
- **Gradient Clipping**: Prevents exploding gradients

## 🛠️ Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Dependencies

```bash
pip install -r requirements.txt
```

### NLTK Data Setup

The project requires specific NLTK resources:

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
```

## 📖 Usage

### Basic Usage

```python
from evaluate_model import SentimentAnalyzer

# Load trained model
analyzer = SentimentAnalyzer('models/lstm_sentiment_model.pth')

# Single prediction
result = analyzer.predict("This movie is amazing!")
print(f"Sentiment: {result['sentiment']}, Confidence: {result['confidence']:.2f}")

# Batch prediction
texts = ["Great movie!", "Terrible film.", "It was okay."]
results = analyzer.predict_batch(texts)
```

### Training Custom Model

```python
from train_model import SentimentTrainer

# Initialize trainer
trainer = SentimentTrainer()

# Load your data
# train_texts = ["positive example", "negative example", ...]
# train_labels = [1, 0, ...]

# Train model
model = trainer.train(train_texts, train_labels, epochs=10)
```

## 📁 Project Structure

```
lstm-sentiment-analysis/
├── data_preprocessing.py      # Text preprocessing pipeline
├── lstm_model.py             # LSTM model definitions
├── train_model.py            # Training orchestration
├── evaluate_model.py         # Model evaluation and inference
├── demo.py                   # Complete demonstration
├── test_implementation.py    # Basic validation tests
├── requirements.txt          # Python dependencies
├── sentiment analysis LSTM peerj-cs-07-408.pdf  # Research reference
└── README.md                 # This file
```

## 🎯 Performance

The model achieves excellent performance on the demo dataset with:
- Fast training (< 1 minute on CPU)
- Comprehensive evaluation metrics
- Confidence scoring for predictions
- Visualization of training progress

## 🔬 Technical Details

- **Framework**: PyTorch 2.0+
- **Architecture**: Bidirectional LSTM with embedding layers
- **Optimization**: Adam optimizer with learning rate scheduling
- **Regularization**: Dropout and early stopping
- **Evaluation**: ROC-AUC, precision, recall, F1-score

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

- Research paper included: `sentiment analysis LSTM peerj-cs-07-408.pdf`
- [PyTorch Documentation](https://pytorch.org/docs/)
- [NLTK Documentation](https://www.nltk.org/)

## 🙏 Acknowledgments

- Built with PyTorch and NLTK
- Inspired by modern NLP research and best practices
- Designed for educational and research purposes