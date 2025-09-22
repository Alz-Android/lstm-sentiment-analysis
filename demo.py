"""
Sentiment Analysis Demo Script
This script demonstrates the complete sentiment analysis pipeline.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd
import json
from datetime import datetime
import os

# Import our modules
from data_preprocessing import TextPreprocessor, load_sample_data
from lstm_model import LSTMSentimentModel, save_model
from train_model import SentimentTrainer, create_data_loaders, create_train_test_split
from evaluate_model import SentimentAnalyzer, ModelEvaluator, create_prediction_report

def demo_preprocessing():
    """Demonstrate text preprocessing capabilities."""
    print("=" * 60)
    print("DEMO 1: TEXT PREPROCESSING")
    print("=" * 60)
    
    # Sample texts with various challenges
    sample_texts = [
        "I LOVE this movie!!! It's absolutely AMAZING! 😊",
        "This product is terrible... I hate it so much 😠",
        "The weather is okay today, nothing special.",
        "Best purchase EVER! Highly recommend to everyone!!!",
        "Worst experience... Never coming back. Very disappointed.",
        "http://example.com Check this out! @user #awesome",
        "Mixed feelings... Some parts good, others bad.",
        "!!URGENT!! This is the WORST service I've ever seen!!!",
    ]
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(max_vocab_size=100, max_sequence_length=20)
    
    print("Sample Text Preprocessing:")
    print("-" * 40)
    
    for i, text in enumerate(sample_texts[:3], 1):
        print(f"{i}. Original: {text}")
        cleaned = preprocessor.clean_text(text)
        print(f"   Cleaned:  {cleaned}")
        tokens = preprocessor.tokenize_text(cleaned)
        print(f"   Tokens:   {tokens}")
        print()
    
    # Build vocabulary and show preprocessing pipeline
    print("Building vocabulary...")
    preprocessor.build_vocabulary(sample_texts)
    
    # Show vocabulary
    print(f"Vocabulary size: {preprocessor.vocab_size}")
    print("Sample vocabulary items:")
    vocab_items = list(preprocessor.vocab_to_int.items())[:10]
    for word, idx in vocab_items:
        print(f"  {word}: {idx}")
    
    # Demonstrate full preprocessing
    print("\nFull preprocessing pipeline:")
    X = preprocessor.preprocess_texts(sample_texts)
    print(f"Processed shape: {X.shape}")
    print(f"Sample sequence: {X[0]}")
    
    return preprocessor, sample_texts

def demo_model_training():
    """Demonstrate model training process."""
    print("\n" + "=" * 60)
    print("DEMO 2: MODEL TRAINING")
    print("=" * 60)
    
    # Load sample data
    texts, labels = load_sample_data()
    print(f"Loaded {len(texts)} samples")
    print(f"Positive samples: {sum(labels)}")
    print(f"Negative samples: {len(labels) - sum(labels)}")
    
    # Split data
    X_train_texts, X_val_texts, y_train, y_val = create_train_test_split(
        texts, labels, test_size=0.3, random_state=42
    )
    
    print(f"Training samples: {len(X_train_texts)}")
    print(f"Validation samples: {len(X_val_texts)}")
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(max_vocab_size=500, max_sequence_length=30)
    preprocessor.build_vocabulary(X_train_texts)
    
    # Preprocess data
    X_train, _ = preprocessor.preprocess_texts(X_train_texts, y_train)
    X_val, _ = preprocessor.preprocess_texts(X_val_texts, y_val)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        X_train, y_train, X_val, y_val, batch_size=4
    )
    
    # Create model
    print("\nCreating LSTM model...")
    model = LSTMSentimentModel(
        vocab_size=preprocessor.vocab_size,
        embedding_dim=32,
        hidden_dim=32,
        output_dim=1,
        n_layers=1,
        dropout=0.3
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = SentimentTrainer(model, device=device, learning_rate=0.01)
    
    print(f"\nTraining on device: {device}")
    history = trainer.train(
        train_loader, val_loader, 
        epochs=5,  # Quick training for demo
        save_path='demo_sentiment_model.pth'
    )
    
    # Plot training history
    trainer.plot_training_history(history)
    
    return model, preprocessor, history

def demo_sentiment_prediction():
    """Demonstrate sentiment prediction."""
    print("\n" + "=" * 60)
    print("DEMO 3: SENTIMENT PREDICTION")
    print("=" * 60)
    
    # Train a quick model
    model, preprocessor, _ = demo_model_training()
    
    # Create analyzer
    analyzer = SentimentAnalyzer('demo_sentiment_model.pth')
    analyzer.set_preprocessor(preprocessor)
    
    # Test texts covering various scenarios
    test_texts = [
        # Clear positive examples
        "I absolutely love this product! It's amazing!",
        "Fantastic service, highly recommend to everyone!",
        "Best movie I've ever seen, truly wonderful!",
        
        # Clear negative examples
        "This is the worst thing I've ever bought.",
        "Terrible experience, very disappointed.",
        "I hate this product, complete waste of money.",
        
        # Neutral/Mixed examples
        "The product is okay, nothing special.",
        "It's alright, could be better though.",
        "Mixed feelings about this purchase.",
        
        # Challenging examples
        "Not bad, but not great either.",
        "I don't love it, but I don't hate it.",
        "Could be worse, I suppose.",
    ]
    
    print("Sentiment Predictions:")
    print("-" * 40)
    
    results = []
    for i, text in enumerate(test_texts, 1):
        result = analyzer.predict_single(text)
        results.append(result)
        
        print(f"{i:2d}. Text: {text}")
        print(f"    Sentiment: {result['sentiment']}")
        print(f"    Confidence: {result['confidence']:.3f}")
        print(f"    Probability: {result['probability']:.3f}")
        print()
    
    # Create visualization
    create_prediction_visualization(results)
    
    # Create prediction report
    report = create_prediction_report(analyzer, test_texts, 'demo_prediction_report.json')
    
    print(f"Prediction Summary:")
    print(f"Total predictions: {report['total_predictions']}")
    print(f"Positive predictions: {report['positive_predictions']}")
    print(f"Negative predictions: {report['negative_predictions']}")
    print(f"Average confidence: {report['average_confidence']:.3f}")
    
    return analyzer, results

def create_prediction_visualization(results):
    """Create visualizations for prediction results."""
    print("Creating prediction visualizations...")
    
    # Extract data
    sentiments = [r['sentiment'] for r in results]
    confidences = [r['confidence'] for r in results]
    probabilities = [r['probability'] for r in results]
    texts = [r['text'][:50] + '...' if len(r['text']) > 50 else r['text'] for r in results]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Sentiment distribution
    sentiment_counts = pd.Series(sentiments).value_counts()
    axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
    axes[0, 0].set_title('Sentiment Distribution')
    
    # 2. Confidence distribution
    axes[0, 1].hist(confidences, bins=10, color='lightblue', edgecolor='black')
    axes[0, 1].set_title('Confidence Score Distribution')
    axes[0, 1].set_xlabel('Confidence Score')
    axes[0, 1].set_ylabel('Frequency')
    
    # 3. Probability vs Confidence scatter
    colors = ['red' if s == 'Negative' else 'green' for s in sentiments]
    axes[1, 0].scatter(probabilities, confidences, c=colors, alpha=0.7)
    axes[1, 0].set_title('Probability vs Confidence')
    axes[1, 0].set_xlabel('Probability')
    axes[1, 0].set_ylabel('Confidence')
    axes[1, 0].axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
    
    # 4. Confidence by prediction
    pos_conf = [c for c, s in zip(confidences, sentiments) if s == 'Positive']
    neg_conf = [c for c, s in zip(confidences, sentiments) if s == 'Negative']
    
    axes[1, 1].boxplot([pos_conf, neg_conf], labels=['Positive', 'Negative'])
    axes[1, 1].set_title('Confidence by Sentiment')
    axes[1, 1].set_ylabel('Confidence Score')
    
    plt.tight_layout()
    plt.savefig('prediction_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def demo_model_evaluation():
    """Demonstrate comprehensive model evaluation."""
    print("\n" + "=" * 60)
    print("DEMO 4: MODEL EVALUATION")
    print("=" * 60)
    
    # Load larger sample data for evaluation
    texts, labels = load_sample_data()
    
    # Add more diverse examples for better evaluation
    additional_texts = [
        "Okay product, does what it says",
        "Not impressed, expected better quality",
        "Satisfied with the purchase",
        "Could be improved but acceptable",
        "Outstanding quality and service!",
        "Disappointing results, not recommended",
        "Good value for money",
        "Mediocre performance, nothing special"
    ]
    additional_labels = [1, 0, 1, 0, 1, 0, 1, 0]  # Mixed labels
    
    texts.extend(additional_texts)
    labels.extend(additional_labels)
    
    # Split data
    X_train_texts, X_test_texts, y_train, y_test = create_train_test_split(
        texts, labels, test_size=0.4, random_state=42
    )
    
    # Initialize preprocessor and model
    preprocessor = TextPreprocessor(max_vocab_size=500, max_sequence_length=30)
    preprocessor.build_vocabulary(X_train_texts)
    
    # Quick model training
    X_train, _ = preprocessor.preprocess_texts(X_train_texts, y_train)
    X_val, _ = preprocessor.preprocess_texts(X_test_texts, y_test)
    
    train_loader, val_loader = create_data_loaders(
        X_train, y_train, X_val, y_test, batch_size=4
    )
    
    model = LSTMSentimentModel(
        vocab_size=preprocessor.vocab_size,
        embedding_dim=32,
        hidden_dim=32,
        output_dim=1,
        n_layers=1,
        dropout=0.3
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = SentimentTrainer(model, device=device, learning_rate=0.01)
    
    print("Training model for evaluation...")
    history = trainer.train(train_loader, val_loader, epochs=5, save_path='eval_model.pth')
    
    # Evaluate model
    evaluator = ModelEvaluator(model, preprocessor, device)
    results = evaluator.evaluate_dataset(X_test_texts, y_test)
    
    print("\nEvaluation Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"ROC AUC: {results['roc_auc']:.4f}")
    print(f"PR AUC: {results['pr_auc']:.4f}")
    
    # Plot evaluation results
    evaluator.plot_evaluation_results(results)
    
    return results

def interactive_demo():
    """Interactive sentiment analysis demo."""
    print("\n" + "=" * 60)
    print("DEMO 5: INTERACTIVE SENTIMENT ANALYSIS")
    print("=" * 60)
    
    try:
        # Quick training
        model, preprocessor, _ = demo_model_training()
        analyzer = SentimentAnalyzer('demo_sentiment_model.pth')
        analyzer.set_preprocessor(preprocessor)
        
        print("\nInteractive Sentiment Analysis")
        print("Enter text to analyze (type 'quit' to exit):")
        print("-" * 40)
        
        while True:
            user_input = input("\nYour text: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Thanks for using the sentiment analyzer!")
                break
            
            if not user_input:
                print("Please enter some text.")
                continue
            
            try:
                result = analyzer.predict_single(user_input)
                
                print(f"Sentiment: {result['sentiment']}")
                print(f"Confidence: {result['confidence']:.3f}")
                print(f"Probability: {result['probability']:.3f}")
                
                # Add interpretation
                if result['confidence'] > 0.8:
                    confidence_level = "Very High"
                elif result['confidence'] > 0.6:
                    confidence_level = "High"
                elif result['confidence'] > 0.4:
                    confidence_level = "Medium"
                else:
                    confidence_level = "Low"
                
                print(f"Confidence Level: {confidence_level}")
                
            except Exception as e:
                print(f"Error analyzing text: {e}")
                
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"Error in interactive demo: {e}")

def create_demo_summary():
    """Create a summary of all demo results."""
    print("\n" + "=" * 60)
    print("DEMO SUMMARY")
    print("=" * 60)
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "demos_completed": [
            "Text Preprocessing",
            "Model Training", 
            "Sentiment Prediction",
            "Model Evaluation",
            "Interactive Demo"
        ],
        "files_created": [
            "demo_sentiment_model.pth",
            "demo_prediction_report.json",
            "prediction_visualization.png",
            "training_history.png",
            "evaluation_results.png"
        ],
        "key_features_demonstrated": [
            "Text cleaning and tokenization",
            "Vocabulary building",
            "LSTM model architecture",
            "Training with validation",
            "Batch prediction",
            "Confidence scoring",
            "Comprehensive evaluation",
            "Visualization generation"
        ]
    }
    
    print("Successfully demonstrated:")
    for feature in summary["key_features_demonstrated"]:
        print(f"  ✓ {feature}")
    
    print(f"\nFiles created:")
    for file in summary["files_created"]:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (not found)")
    
    # Save summary
    with open('demo_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nDemo summary saved to: demo_summary.json")
    return summary

def main():
    """Run the complete demo."""
    print("LSTM SENTIMENT ANALYSIS - COMPLETE DEMO")
    print("=" * 60)
    print("This demo will showcase all components of the sentiment analysis system.")
    print("Please wait while we run through each demonstration...")
    
    try:
        # Run all demos
        preprocessor, sample_texts = demo_preprocessing()
        model, trained_preprocessor, history = demo_model_training()
        analyzer, results = demo_sentiment_prediction()
        eval_results = demo_model_evaluation()
        
        # Create summary
        summary = create_demo_summary()
        
        print("\n" + "=" * 60)
        print("ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Check the generated files for detailed results and visualizations.")
        
        # Optionally run interactive demo
        response = input("\nWould you like to try the interactive demo? (y/n): ")
        if response.lower().startswith('y'):
            interactive_demo()
        
        return summary
        
    except Exception as e:
        print(f"Error in demo: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    summary = main()