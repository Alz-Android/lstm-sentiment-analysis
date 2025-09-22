"""
Model evaluation and prediction module for sentiment analysis.
This module provides comprehensive evaluation tools and prediction functions.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_curve, auc, precision_recall_curve
)
from wordcloud import WordCloud
import json
import os
from datetime import datetime

from lstm_model import LSTMSentimentModel, AdvancedLSTMSentimentModel, load_model, get_sequence_lengths
from data_preprocessing import TextPreprocessor

class SentimentAnalyzer:
    """
    Complete sentiment analysis system for evaluation and prediction.
    """
    
    def __init__(self, model_path, preprocessor=None, device='cpu'):
        """
        Initialize the sentiment analyzer.
        
        Args:
            model_path (str): Path to trained model
            preprocessor (TextPreprocessor, optional): Text preprocessor
            device (str): Device to use for inference
        """
        self.device = device
        self.model_path = model_path
        
        # Load model
        self.model, self.checkpoint = load_model(model_path, device=device)
        self.model.eval()
        
        # Set preprocessor
        self.preprocessor = preprocessor
        
        # Prediction cache
        self.prediction_cache = {}
        
    def set_preprocessor(self, preprocessor):
        """Set the text preprocessor."""
        self.preprocessor = preprocessor
    
    def predict_single(self, text):
        """
        Predict sentiment for a single text.
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Prediction results
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not set. Use set_preprocessor() first.")
        
        # Check cache
        if text in self.prediction_cache:
            return self.prediction_cache[text]
        
        # Preprocess text
        sequence = self.preprocessor.text_to_sequence(text)
        padded_sequence = self.preprocessor.pad_sequences([sequence])
        
        # Convert to tensor
        input_tensor = torch.tensor(padded_sequence, dtype=torch.long).to(self.device)
        length_tensor = get_sequence_lengths(input_tensor).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(input_tensor, length_tensor)
            probability = torch.sigmoid(logits).item()
            prediction = int(probability > 0.5)
        
        result = {
            'text': text,
            'prediction': prediction,
            'probability': probability,
            'confidence': abs(probability - 0.5) * 2,  # Confidence score
            'sentiment': 'Positive' if prediction == 1 else 'Negative'
        }
        
        # Cache result
        self.prediction_cache[text] = result
        
        return result
    
    def predict_batch(self, texts, batch_size=32):
        """
        Predict sentiment for multiple texts.
        
        Args:
            texts (list): List of input texts
            batch_size (int): Batch size for processing
            
        Returns:
            list: List of prediction results
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not set. Use set_preprocessor() first.")
        
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Preprocess batch
            sequences = [self.preprocessor.text_to_sequence(text) for text in batch_texts]
            padded_sequences = self.preprocessor.pad_sequences(sequences)
            
            # Convert to tensors
            input_tensor = torch.tensor(padded_sequences, dtype=torch.long).to(self.device)
            length_tensor = get_sequence_lengths(input_tensor).to(self.device)
            
            # Predict
            with torch.no_grad():
                logits = self.model(input_tensor, length_tensor)
                probabilities = torch.sigmoid(logits).cpu().numpy().flatten()
                predictions = (probabilities > 0.5).astype(int)
            
            # Create results
            for j, text in enumerate(batch_texts):
                prob = probabilities[j]
                pred = predictions[j]
                result = {
                    'text': text,
                    'prediction': pred,
                    'probability': prob,
                    'confidence': abs(prob - 0.5) * 2,
                    'sentiment': 'Positive' if pred == 1 else 'Negative'
                }
                results.append(result)
        
        return results

def create_prediction_report(analyzer, test_texts, output_file='prediction_report.json'):
    """
    Create a detailed prediction report.
    
    Args:
        analyzer (SentimentAnalyzer): Sentiment analyzer
        test_texts (list): List of test texts
        output_file (str): Output file path
        
    Returns:
        dict: Prediction report
    """
    print("Creating prediction report...")
    
    # Get predictions
    predictions = analyzer.predict_batch(test_texts)
    
    # Analyze results
    positive_count = sum(1 for p in predictions if p['prediction'] == 1)
    negative_count = len(predictions) - positive_count
    
    # Find high and low confidence predictions
    sorted_predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
    high_confidence = sorted_predictions[:5]
    low_confidence = sorted_predictions[-5:]
    
    # Calculate statistics
    avg_confidence = np.mean([p['confidence'] for p in predictions])
    avg_probability = np.mean([p['probability'] for p in predictions])
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_predictions': len(predictions),
        'positive_predictions': positive_count,
        'negative_predictions': negative_count,
        'positive_ratio': positive_count / len(predictions),
        'average_confidence': avg_confidence,
        'average_probability': avg_probability,
        'high_confidence_predictions': high_confidence,
        'low_confidence_predictions': low_confidence,
        'all_predictions': predictions
    }
    
    # Save report
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"Prediction report saved to {output_file}")
    return report

if __name__ == "__main__":
    # Demo of evaluation module
    print("Sentiment Analysis Evaluation Demo")
    print("=" * 50)
    
    # Note: This requires a trained model to be available
    try:
        # Load model and preprocessor (these would need to be available)
        from train_model import main as train_main
        
        # Train model first
        print("Training model...")
        model, preprocessor, history, metrics = train_main()
        
        # Create analyzer
        analyzer = SentimentAnalyzer('sentiment_model.pth')
        analyzer.set_preprocessor(preprocessor)
        
        # Test predictions
        test_texts = [
            "This movie is absolutely fantastic!",
            "I hate this product, it's terrible.",
            "The weather is okay today.",
            "Amazing experience, highly recommended!",
            "Worst service ever, very disappointed."
        ]
        
        print("\nTesting predictions...")
        for text in test_texts:
            result = analyzer.predict_single(text)
            print(f"Text: {text}")
            print(f"Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.3f})")
            print("-" * 50)
        
        # Create prediction report
        report = create_prediction_report(analyzer, test_texts)
        print(f"\nPrediction Summary:")
        print(f"Total predictions: {report['total_predictions']}")
        print(f"Positive: {report['positive_predictions']}")
        print(f"Negative: {report['negative_predictions']}")
        print(f"Average confidence: {report['average_confidence']:.3f}")
        
    except Exception as e:
        print(f"Demo requires trained model: {e}")
        print("Run train_model.py first to train a model.")