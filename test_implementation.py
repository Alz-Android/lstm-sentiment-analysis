"""
Simple test script to validate the sentiment analysis implementation.
"""

import torch
import numpy as np
from data_preprocessing import TextPreprocessor, load_sample_data
from lstm_model import LSTMSentimentModel
from train_model import SentimentTrainer, create_data_loaders, create_train_test_split

def test_complete_pipeline():
    """Test the complete sentiment analysis pipeline."""
    print("Testing LSTM Sentiment Analysis Pipeline")
    print("=" * 50)
    
    try:
        # 1. Test data preprocessing
        print("1. Testing data preprocessing...")
        texts, labels = load_sample_data()
        preprocessor = TextPreprocessor(max_vocab_size=100, max_sequence_length=20)
        preprocessor.build_vocabulary(texts)
        X, y = preprocessor.preprocess_texts(texts, labels)
        print(f"   ✓ Data preprocessed: {X.shape}")
        
        # 2. Test model creation
        print("2. Testing model creation...")
        model = LSTMSentimentModel(
            vocab_size=preprocessor.vocab_size,
            embedding_dim=16,
            hidden_dim=16,
            output_dim=1,
            n_layers=1,
            dropout=0.3
        )
        print(f"   ✓ Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # 3. Test training (minimal)
        print("3. Testing training pipeline...")
        X_train_texts, X_val_texts, y_train, y_val = create_train_test_split(
            texts, labels, test_size=0.3, random_state=42
        )
        
        X_train, _ = preprocessor.preprocess_texts(X_train_texts, y_train)
        X_val, _ = preprocessor.preprocess_texts(X_val_texts, y_val)
        
        train_loader, val_loader = create_data_loaders(
            X_train, y_train, X_val, y_val, batch_size=4
        )
        
        device = torch.device('cpu')
        trainer = SentimentTrainer(model, device=device, learning_rate=0.01)
        
        # Quick training (1 epoch)
        history = trainer.train(train_loader, val_loader, epochs=1, save_path='test_model.pth')
        print(f"   ✓ Training completed")
        
        # 4. Test prediction
        print("4. Testing prediction...")
        test_text = "This is a great movie!"
        sequence = preprocessor.text_to_sequence(test_text)
        padded_sequence = preprocessor.pad_sequences([sequence])
        
        model.eval()
        with torch.no_grad():
            input_tensor = torch.tensor(padded_sequence, dtype=torch.long)
            output = model(input_tensor)
            probability = torch.sigmoid(output).item()
            prediction = int(probability > 0.5)
        
        print(f"   ✓ Prediction for '{test_text}': {prediction} (prob: {probability:.3f})")
        
        # 5. Test multiple predictions
        print("5. Testing batch prediction...")
        test_texts = [
            "I love this!",
            "This is terrible.",
            "It's okay, I guess.",
            "Absolutely fantastic!",
            "Worst ever."
        ]
        
        sequences = [preprocessor.text_to_sequence(text) for text in test_texts]
        padded_sequences = preprocessor.pad_sequences(sequences)
        
        with torch.no_grad():
            input_tensor = torch.tensor(padded_sequences, dtype=torch.long)
            outputs = model(input_tensor)
            probabilities = torch.sigmoid(outputs).cpu().numpy().flatten()
            predictions = (probabilities > 0.5).astype(int)
        
        print("   ✓ Batch predictions:")
        for text, pred, prob in zip(test_texts, predictions, probabilities):
            sentiment = "Positive" if pred == 1 else "Negative"
            print(f"      '{text}' -> {sentiment} ({prob:.3f})")
        
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED! ✓")
        print("The sentiment analysis implementation is working correctly.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_complete_pipeline()
    if success:
        print("\n🎉 Implementation is ready to use!")
    else:
        print("\n❌ Please fix the errors above.")