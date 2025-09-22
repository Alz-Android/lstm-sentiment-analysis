"""
LSTM Sentiment Analysis Model Implementation
This module contains the LSTM model architecture for sentiment analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LSTMSentimentModel(nn.Module):
    """
    LSTM-based sentiment analysis model.
    
    Architecture:
    1. Embedding layer: Converts word indices to dense vectors
    2. LSTM layer: Processes sequential information
    3. Dropout layer: Prevents overfitting
    4. Fully connected layer: Maps to output classes
    """
    
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, 
                 output_dim=1, n_layers=2, dropout=0.3, pad_idx=0):
        """
        Initialize the LSTM sentiment model.
        
        Args:
            vocab_size (int): Size of vocabulary
            embedding_dim (int): Dimension of word embeddings
            hidden_dim (int): Hidden dimension of LSTM
            output_dim (int): Output dimension (1 for binary classification)
            n_layers (int): Number of LSTM layers
            dropout (float): Dropout probability
            pad_idx (int): Padding token index
        """
        super(LSTMSentimentModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                           dropout=dropout if n_layers > 1 else 0,
                           batch_first=True, bidirectional=True)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Fully connected layer (bidirectional LSTM doubles the hidden size)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize embeddings
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[(n//4):(n//2)].fill_(1.0)
        
        # Initialize fully connected layer
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)
    
    def forward(self, x, lengths=None):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length)
            lengths (torch.Tensor, optional): Actual lengths of sequences
            
        Returns:
            torch.Tensor: Output predictions
        """
        batch_size = x.size(0)
        
        # Embedding layer
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        
        # Pack sequences if lengths are provided
        if lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False
            )
        
        # LSTM layer
        lstm_out, (hidden, _) = self.lstm(embedded)
        
        # Unpack sequences if they were packed
        if lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True
            )
        
        # Use the last output from both directions
        # For bidirectional LSTM, we concatenate the last outputs from both directions
        if lengths is not None:
            # Get the actual last output for each sequence
            batch_size, max_len, hidden_size = lstm_out.size()
            idx = (lengths - 1).unsqueeze(1).unsqueeze(1).expand(batch_size, 1, hidden_size)
            last_output = lstm_out.gather(1, idx.long()).squeeze(1)
        else:
            # Use the last output
            last_output = lstm_out[:, -1, :]
        
        # Apply dropout
        last_output = self.dropout_layer(last_output)
        
        # Fully connected layer
        output = self.fc(last_output)
        
        return output
    
    def predict_sentiment(self, x, lengths=None):
        """
        Predict sentiment with probabilities.
        
        Args:
            x (torch.Tensor): Input tensor
            lengths (torch.Tensor, optional): Actual lengths of sequences
            
        Returns:
            dict: Dictionary with predictions and probabilities
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, lengths)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > 0.5).long()
            
            return {
                'predictions': predictions,
                'probabilities': probabilities,
                'logits': logits
            }

class AdvancedLSTMSentimentModel(nn.Module):
    """
    Advanced LSTM model with attention mechanism and additional features.
    """
    
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128,
                 output_dim=1, n_layers=2, dropout=0.3, pad_idx=0,
                 use_attention=True):
        """
        Initialize advanced LSTM model.
        
        Args:
            vocab_size (int): Size of vocabulary
            embedding_dim (int): Dimension of word embeddings
            hidden_dim (int): Hidden dimension of LSTM
            output_dim (int): Output dimension
            n_layers (int): Number of LSTM layers
            dropout (float): Dropout probability
            pad_idx (int): Padding token index
            use_attention (bool): Whether to use attention mechanism
        """
        super(AdvancedLSTMSentimentModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.use_attention = use_attention
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                           dropout=dropout if n_layers > 1 else 0,
                           batch_first=True, bidirectional=True)
        
        # Attention mechanism
        if use_attention:
            self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Fully connected layers
        fc_input_dim = hidden_dim * 2
        self.fc1 = nn.Linear(fc_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize embeddings
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
                n = param.size(0)
                param.data[(n//4):(n//2)].fill_(1.0)
        
        # Initialize fully connected layers
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.constant_(self.fc2.bias, 0.0)
        
        # Initialize attention layer
        if self.use_attention:
            nn.init.xavier_uniform_(self.attention.weight)
            nn.init.constant_(self.attention.bias, 0.0)
    
    def attention_mechanism(self, lstm_outputs, mask=None):
        """
        Apply attention mechanism to LSTM outputs.
        
        Args:
            lstm_outputs (torch.Tensor): LSTM outputs
            mask (torch.Tensor, optional): Padding mask
            
        Returns:
            torch.Tensor: Attention-weighted output
        """
        # Compute attention scores
        attention_scores = self.attention(lstm_outputs).squeeze(-1)
        
        # Apply mask if provided
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Apply attention weights to LSTM outputs
        attended_output = torch.sum(lstm_outputs * attention_weights.unsqueeze(-1), dim=1)
        
        return attended_output, attention_weights
    
    def forward(self, x, lengths=None):
        """
        Forward pass of the advanced model.
        
        Args:
            x (torch.Tensor): Input tensor
            lengths (torch.Tensor, optional): Actual lengths of sequences
            
        Returns:
            torch.Tensor: Output predictions
        """
        batch_size, seq_length = x.size()
        
        # Create mask for padding
        if lengths is not None:
            mask = torch.arange(seq_length).expand(batch_size, seq_length) < lengths.unsqueeze(1)
            mask = mask.to(x.device)
        else:
            mask = None
        
        # Embedding layer
        embedded = self.embedding(x)
        
        # LSTM layer
        lstm_out, _ = self.lstm(embedded)
        
        # Apply attention or use last output
        if self.use_attention:
            attended_output, _ = self.attention_mechanism(lstm_out, mask)
        else:
            if lengths is not None:
                # Get the actual last output for each sequence
                idx = (lengths - 1).unsqueeze(1).unsqueeze(1).expand(batch_size, 1, lstm_out.size(-1))
                attended_output = lstm_out.gather(1, idx.long()).squeeze(1)
            else:
                attended_output = lstm_out[:, -1, :]
        
        # Apply dropout
        attended_output = self.dropout_layer(attended_output)
        
        # First fully connected layer with batch norm and activation
        fc1_out = self.fc1(attended_output)
        fc1_out = self.batch_norm(fc1_out)
        fc1_out = F.relu(fc1_out)
        fc1_out = self.dropout_layer(fc1_out)
        
        # Second fully connected layer
        output = self.fc2(fc1_out)
        
        return output

def get_sequence_lengths(sequences, pad_token=0):
    """
    Get actual lengths of sequences (excluding padding).
    
    Args:
        sequences (torch.Tensor): Padded sequences
        pad_token (int): Padding token value
        
    Returns:
        torch.Tensor: Actual sequence lengths
    """
    mask = sequences != pad_token
    lengths = mask.sum(dim=1)
    return lengths

def save_model(model, filepath, optimizer=None, epoch=None, loss=None):
    """
    Save model checkpoint.
    
    Args:
        model (nn.Module): PyTorch model
        filepath (str): Path to save the model
        optimizer (torch.optim.Optimizer, optional): Optimizer state
        epoch (int, optional): Current epoch
        loss (float, optional): Current loss
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'vocab_size': model.vocab_size,
            'embedding_dim': model.embedding_dim,
            'hidden_dim': model.hidden_dim,
            'output_dim': model.output_dim,
            'n_layers': model.n_layers,
            'dropout': model.dropout
        }
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if loss is not None:
        checkpoint['loss'] = loss
    
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath, model_class=LSTMSentimentModel, device='cpu'):
    """
    Load model from checkpoint.
    
    Args:
        filepath (str): Path to model checkpoint
        model_class (nn.Module): Model class to instantiate
        device (str): Device to load model on
        
    Returns:
        nn.Module: Loaded model
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    # Create model with saved configuration
    model = model_class(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"Model loaded from {filepath}")
    return model, checkpoint

if __name__ == "__main__":
    # Demo of the model
    print("LSTM Sentiment Analysis Model Demo")
    
    # Model parameters
    vocab_size = 1000
    embedding_dim = 100
    hidden_dim = 128
    output_dim = 1
    
    # Create model
    model = LSTMSentimentModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test with dummy input
    batch_size = 4
    seq_length = 20
    
    # Create dummy input
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_length))
    dummy_lengths = torch.tensor([15, 20, 10, 18])
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input, dummy_lengths)
        print(f"Output shape: {output.shape}")
        print(f"Sample output: {output[:2].squeeze()}")
        
        # Test prediction
        predictions = model.predict_sentiment(dummy_input, dummy_lengths)
        print(f"Predictions: {predictions['predictions'].squeeze()}")
        print(f"Probabilities: {predictions['probabilities'].squeeze()}")