"""
Data preprocessing module for sentiment analysis.
This module handles text cleaning, tokenization, and vocabulary building.
"""

import re
import string
import nltk
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class TextPreprocessor:
    """Text preprocessing class for sentiment analysis."""
    
    def __init__(self, max_vocab_size=10000, max_sequence_length=100):
        """
        Initialize the preprocessor.
        
        Args:
            max_vocab_size (int): Maximum vocabulary size
            max_sequence_length (int): Maximum sequence length for padding
        """
        self.max_vocab_size = max_vocab_size
        self.max_sequence_length = max_sequence_length
        self.vocab_to_int = {}
        self.int_to_vocab = {}
        self.vocab_size = 0
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text):
        """
        Clean and normalize text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove punctuation except for emoticons
        text = re.sub(r'[^\w\s\:\)\(\;\-\!\?]', '', text)
        
        return text
    
    def tokenize_text(self, text):
        """
        Tokenize text and remove stopwords.
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of tokens
        """
        tokens = word_tokenize(text)
        
        # Remove stopwords and single characters
        tokens = [token for token in tokens 
                 if token not in self.stop_words and len(token) > 1]
        
        return tokens
    
    def build_vocabulary(self, texts):
        """
        Build vocabulary from texts.
        
        Args:
            texts (list): List of texts
        """
        word_counts = Counter()
        
        for text in texts:
            cleaned_text = self.clean_text(text)
            tokens = self.tokenize_text(cleaned_text)
            word_counts.update(tokens)
        
        # Get most common words
        most_common = word_counts.most_common(self.max_vocab_size - 2)
        
        # Build vocabulary mappings
        self.vocab_to_int = {
            '<PAD>': 0,  # Padding token
            '<UNK>': 1   # Unknown token
        }
        
        for i, (word, _) in enumerate(most_common):
            self.vocab_to_int[word] = i + 2
            
        self.int_to_vocab = {v: k for k, v in self.vocab_to_int.items()}
        self.vocab_size = len(self.vocab_to_int)
        
        print(f"Vocabulary built with {self.vocab_size} words")
    
    def text_to_sequence(self, text):
        """
        Convert text to sequence of integers.
        
        Args:
            text (str): Input text
            
        Returns:
            list: Sequence of integers
        """
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize_text(cleaned_text)
        
        sequence = []
        for token in tokens:
            if token in self.vocab_to_int:
                sequence.append(self.vocab_to_int[token])
            else:
                sequence.append(self.vocab_to_int['<UNK>'])
                
        return sequence
    
    def pad_sequences(self, sequences):
        """
        Pad sequences to same length.
        
        Args:
            sequences (list): List of sequences
            
        Returns:
            np.array: Padded sequences
        """
        padded_sequences = np.zeros((len(sequences), self.max_sequence_length), dtype=int)
        
        for i, sequence in enumerate(sequences):
            if len(sequence) <= self.max_sequence_length:
                padded_sequences[i, :len(sequence)] = sequence
            else:
                padded_sequences[i, :] = sequence[:self.max_sequence_length]
                
        return padded_sequences
    
    def preprocess_texts(self, texts, labels=None):
        """
        Complete preprocessing pipeline.
        
        Args:
            texts (list): List of texts
            labels (list, optional): List of labels
            
        Returns:
            tuple: Processed sequences and labels (if provided)
        """
        # Convert texts to sequences
        sequences = [self.text_to_sequence(text) for text in texts]
        
        # Pad sequences
        padded_sequences = self.pad_sequences(sequences)
        
        if labels is not None:
            return padded_sequences, np.array(labels)
        else:
            return padded_sequences

def load_sample_data():
    """
    Load sample sentiment analysis data.
    Creates a simple dataset for demonstration.
    
    Returns:
        tuple: (texts, labels)
    """
    # Sample positive texts
    positive_texts = [
        "I love this movie! It's absolutely fantastic.",
        "Great product, highly recommend it to everyone.",
        "Amazing experience, will definitely come back.",
        "Excellent service and friendly staff.",
        "This is the best thing I've ever bought.",
        "Wonderful day, feeling so happy and grateful.",
        "Perfect solution to my problem, thank you!",
        "Outstanding quality and fast delivery.",
        "Brilliant performance, exceeded my expectations.",
        "Fantastic restaurant with delicious food."
    ]
    
    # Sample negative texts
    negative_texts = [
        "Terrible movie, waste of time and money.",
        "Poor quality product, very disappointed.",
        "Awful experience, will never come back.",
        "Bad service and rude staff members.",
        "This is the worst purchase I've made.",
        "Horrible day, everything went wrong.",
        "Useless product, doesn't work at all.",
        "Slow delivery and damaged packaging.",
        "Disappointing performance, not worth it.",
        "Disgusting food and dirty restaurant."
    ]
    
    # Combine texts and create labels
    texts = positive_texts + negative_texts
    labels = [1] * len(positive_texts) + [0] * len(negative_texts)
    
    return texts, labels

def create_train_test_split(texts, labels, test_size=0.2, random_state=42):
    """
    Create train-test split.
    
    Args:
        texts (list): List of texts
        labels (list): List of labels
        test_size (float): Test set size ratio
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    return train_test_split(texts, labels, test_size=test_size, 
                          random_state=random_state, stratify=labels)

if __name__ == "__main__":
    # Demo of preprocessing module
    print("Loading sample data...")
    texts, labels = load_sample_data()
    
    print(f"Loaded {len(texts)} samples")
    print(f"Positive samples: {sum(labels)}")
    print(f"Negative samples: {len(labels) - sum(labels)}")
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(max_vocab_size=1000, max_sequence_length=50)
    
    # Build vocabulary
    preprocessor.build_vocabulary(texts)
    
    # Preprocess texts
    X, y = preprocessor.preprocess_texts(texts, labels)
    
    print(f"Preprocessed data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Vocabulary size: {preprocessor.vocab_size}")
    
    # Show example
    print("\nExample preprocessing:")
    print(f"Original text: {texts[0]}")
    print(f"Cleaned text: {preprocessor.clean_text(texts[0])}")
    print(f"Tokens: {preprocessor.tokenize_text(preprocessor.clean_text(texts[0]))}")
    print(f"Sequence: {X[0][:10]}...")  # Show first 10 elements