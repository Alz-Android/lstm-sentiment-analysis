"""
Training script for LSTM sentiment analysis model.
This module handles the complete training pipeline including data loading,
model training, validation, and evaluation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import os
import time

from data_preprocessing import TextPreprocessor, load_sample_data, create_train_test_split
from lstm_model import LSTMSentimentModel, AdvancedLSTMSentimentModel, get_sequence_lengths, save_model

class SentimentTrainer:
    """
    Trainer class for sentiment analysis model.
    """
    
    def __init__(self, model, device='cpu', learning_rate=0.001, weight_decay=1e-5):
        """
        Initialize the trainer.
        
        Args:
            model (nn.Module): PyTorch model to train
            device (str): Device to use for training
            learning_rate (float): Learning rate for optimizer
            weight_decay (float): Weight decay for regularization
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        
        # Loss function for binary classification
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(self, train_loader):
        """
        Train the model for one epoch.
        
        Args:
            train_loader (DataLoader): Training data loader
            
        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            if len(batch) == 3:
                sequences, lengths, labels = batch
                sequences = sequences.to(self.device)
                lengths = lengths.to(self.device)
                labels = labels.to(self.device).float().unsqueeze(1)
            else:
                sequences, labels = batch
                sequences = sequences.to(self.device)
                labels = labels.to(self.device).float().unsqueeze(1)
                lengths = get_sequence_lengths(sequences).to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(sequences, lengths)
            
            # Calculate loss
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item()
            
            # Get predictions
            predictions = torch.sigmoid(outputs) > 0.5
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
            })
        
        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader):
        """
        Validate the model for one epoch.
        
        Args:
            val_loader (DataLoader): Validation data loader
            
        Returns:
            tuple: (average_loss, accuracy, predictions, labels)
        """
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation")
            
            for batch in progress_bar:
                if len(batch) == 3:
                    sequences, lengths, labels = batch
                    sequences = sequences.to(self.device)
                    lengths = lengths.to(self.device)
                    labels = labels.to(self.device).float().unsqueeze(1)
                else:
                    sequences, labels = batch
                    sequences = sequences.to(self.device)
                    labels = labels.to(self.device).float().unsqueeze(1)
                    lengths = get_sequence_lengths(sequences).to(self.device)
                
                # Forward pass
                outputs = self.model(sequences, lengths)
                
                # Calculate loss
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                # Get predictions and probabilities
                probabilities = torch.sigmoid(outputs)
                predictions = probabilities > 0.5
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return avg_loss, accuracy, all_predictions, all_labels, all_probabilities
    
    def train(self, train_loader, val_loader, epochs=10, save_path='best_model.pth'):
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            epochs (int): Number of epochs to train
            save_path (str): Path to save the best model
            
        Returns:
            dict: Training history
        """
        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        
        best_val_loss = float('inf')
        best_epoch = 0
        
        start_time = time.time()
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc, val_preds, val_labels, val_probs = self.validate_epoch(val_loader)
            
            # Update learning rate scheduler
            self.scheduler.step(val_loss)
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                save_model(self.model, save_path, self.optimizer, epoch, val_loss)
                print(f"New best model saved! (Val Loss: {val_loss:.4f})")
            
            # Early stopping check
            if epoch - best_epoch > 10:  # Stop if no improvement for 10 epochs
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f} seconds")
        print(f"Best model at epoch {best_epoch+1} with validation loss: {best_val_loss:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
    
    def plot_training_history(self, history=None):
        """
        Plot training history.
        
        Args:
            history (dict, optional): Training history dictionary
        """
        if history is None:
            history = {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'train_accuracies': self.train_accuracies,
                'val_accuracies': self.val_accuracies
            }
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot losses
        ax1.plot(history['train_losses'], label='Train Loss', marker='o')
        ax1.plot(history['val_losses'], label='Validation Loss', marker='s')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracies
        ax2.plot(history['train_accuracies'], label='Train Accuracy', marker='o')
        ax2.plot(history['val_accuracies'], label='Validation Accuracy', marker='s')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def create_data_loaders(X_train, y_train, X_val, y_val, batch_size=32):
    """
    Create PyTorch data loaders.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        batch_size (int): Batch size
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.long)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float)
    X_val_tensor = torch.tensor(X_val, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float)
    
    # Get sequence lengths
    train_lengths = get_sequence_lengths(X_train_tensor)
    val_lengths = get_sequence_lengths(X_val_tensor)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, train_lengths, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, val_lengths, y_val_tensor)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def evaluate_model(model, val_loader, device='cpu'):
    """
    Evaluate model performance with detailed metrics.
    
    Args:
        model (nn.Module): Trained model
        val_loader (DataLoader): Validation data loader
        device (str): Device to use
        
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 3:
                sequences, lengths, labels = batch
                sequences = sequences.to(device)
                lengths = lengths.to(device)
                labels = labels.to(device)
            else:
                sequences, labels = batch
                sequences = sequences.to(device)
                labels = labels.to(device)
                lengths = get_sequence_lengths(sequences).to(device)
            
            outputs = model(sequences, lengths)
            probabilities = torch.sigmoid(outputs)
            predictions = probabilities > 0.5
            
            all_predictions.extend(predictions.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy().flatten())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='binary'
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'predictions': all_predictions,
        'probabilities': all_probabilities,
        'labels': all_labels
    }

def main():
    """Main training function."""
    print("LSTM Sentiment Analysis Training")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    texts, labels = load_sample_data()
    
    # Split data
    X_train_texts, X_val_texts, y_train, y_val = create_train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(max_vocab_size=1000, max_sequence_length=50)
    
    # Build vocabulary on training data
    preprocessor.build_vocabulary(X_train_texts)
    
    # Preprocess data
    X_train, _ = preprocessor.preprocess_texts(X_train_texts, y_train)
    X_val, _ = preprocessor.preprocess_texts(X_val_texts, y_val)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Vocabulary size: {preprocessor.vocab_size}")
    
    # Create data loaders
    print("\n2. Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        X_train, y_train, X_val, y_val, batch_size=8  # Small batch size for demo
    )
    
    # Create model
    print("\n3. Creating model...")
    model = LSTMSentimentModel(
        vocab_size=preprocessor.vocab_size,
        embedding_dim=50,  # Smaller embedding for demo
        hidden_dim=64,     # Smaller hidden dimension for demo
        output_dim=1,
        n_layers=1,        # Single layer for demo
        dropout=0.3
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create trainer
    trainer = SentimentTrainer(model, device=device, learning_rate=0.001)
    
    # Train model
    print("\n4. Training model...")
    history = trainer.train(
        train_loader, val_loader, 
        epochs=10,  # Small number of epochs for demo
        save_path='sentiment_model.pth'
    )
    
    # Plot training history
    print("\n5. Plotting training history...")
    trainer.plot_training_history(history)
    
    # Evaluate model
    print("\n6. Evaluating model...")
    metrics = evaluate_model(model, val_loader, device)
    
    print("\nFinal Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    
    return model, preprocessor, history, metrics

if __name__ == "__main__":
    model, preprocessor, history, metrics = main()