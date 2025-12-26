# train_model.py - Model Training Script

import pandas as pd
import numpy as np
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, LSTM, Conv1D, MaxPooling1D, 
    Dense, Dropout, Bidirectional, concatenate, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import logging

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FakeNewsTrainer:
    """Train fake news detection model"""
    
    def __init__(self, max_words=10000, max_length=500, embedding_dim=100):
        """
        Initialize trainer
        
        Args:
            max_words: Maximum vocabulary size
            max_length: Maximum sequence length
            embedding_dim: Embedding dimension
        """
        self.max_words = max_words
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.tokenizer = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess_data(self, df):
        """
        Preprocess raw data
        
        Args:
            df: DataFrame with 'text' and 'label' columns
            
        Returns:
            Preprocessed text series
        """
        logger.info("Preprocessing data...")
        
        def clean_text(text):
            # Handle missing values
            if pd.isna(text):
                return ""
            
            text = str(text).lower()
            
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            
            # Remove email addresses
            text = re.sub(r'\S+@\S+', '', text)
            
            # Remove special characters and digits
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Tokenize and clean
            tokens = text.split()
            tokens = [
                self.lemmatizer.lemmatize(word) 
                for word in tokens 
                if word not in self.stop_words and len(word) > 2
            ]
            
            return ' '.join(tokens)
        
        df['cleaned_text'] = df['text'].apply(clean_text)
        return df
    
    def load_and_prepare_data(self, file_path):
        """
        Load CSV data and prepare for training
        
        Args:
            file_path: Path to CSV file with 'text' and 'label' columns
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        
        # Handle label encoding
        label_map = {'fake': 1, 'real': 0, 'FAKE': 1, 'REAL': 0, 1: 1, 0: 0}
        df['label'] = df['label'].map(label_map)
        
        # Preprocess
        df = self.preprocess_data(df)
        
        # Remove empty texts
        df = df[df['cleaned_text'].str.len() > 0]
        
        # Split data: 80% train, 10% validation, 10% test
        X_train, X_temp, y_train, y_temp = train_test_split(
            df['cleaned_text'], df['label'], 
            test_size=0.2, random_state=42, stratify=df['label']
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, 
            test_size=0.5, random_state=42, stratify=y_temp
        )
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Tokenize
        logger.info("Tokenizing...")
        self.tokenizer = Tokenizer(num_words=self.max_words)
        self.tokenizer.fit_on_texts(X_train)
        
        # Convert to sequences
        X_train_seq = self.tokenizer.texts_to_sequences(X_train)
        X_val_seq = self.tokenizer.texts_to_sequences(X_val)
        X_test_seq = self.tokenizer.texts_to_sequences(X_test)
        
        # Pad sequences
        X_train_pad = pad_sequences(X_train_seq, maxlen=self.max_length, padding='post')
        X_val_pad = pad_sequences(X_val_seq, maxlen=self.max_length, padding='post')
        X_test_pad = pad_sequences(X_test_seq, maxlen=self.max_length, padding='post')
        
        return (X_train_pad, X_val_pad, X_test_pad, 
                np.array(y_train), np.array(y_val), np.array(y_test))
    
    def build_hybrid_model(self):
        """
        Build hybrid LSTM+CNN model
        
        Returns:
            Compiled Keras model
        """
        logger.info("Building hybrid LSTM+CNN model...")
        
        model = Sequential([
            Embedding(self.max_words, self.embedding_dim, input_length=self.max_length),
            Bidirectional(LSTM(128, return_sequences=True)),
            Dropout(0.3),
            Conv1D(64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=4),
            Bidirectional(LSTM(64)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall()]
        )
        
        logger.info(model.summary())
        return model
    
    def train(self, X_train, X_val, y_train, y_val, epochs=20, batch_size=32):
        """
        Train the model
        
        Args:
            X_train, X_val: Training and validation data
            y_train, y_val: Training and validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Trained model and history
        """
        model = self.build_hybrid_model()
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'models/best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        logger.info("Starting training...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return model, history
    
    def evaluate(self, model, X_test, y_test):
        """
        Evaluate model on test set
        
        Args:
            model: Trained model
            X_test: Test data
            y_test: Test labels
            
        Returns:
            Metrics dictionary
        """
        logger.info("Evaluating model...")
        
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
        logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Test Precision: {metrics['precision']:.4f}")
        logger.info(f"Test Recall: {metrics['recall']:.4f}")
        logger.info(f"Test F1-Score: {metrics['f1']:.4f}")
        
        return metrics
    
    def save_model(self, model, model_path='models/fake_news_model.h5',
                   tokenizer_path='models/tokenizer.pkl'):
        """
        Save trained model and tokenizer
        
        Args:
            model: Trained model
            model_path: Path to save model
            tokenizer_path: Path to save tokenizer
        """
        logger.info(f"Saving model to {model_path}...")
        model.save(model_path)
        
        logger.info(f"Saving tokenizer to {tokenizer_path}...")
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
    
    def plot_history(self, history, save_path='models/training_history.png'):
        """
        Plot training history
        
        Args:
            history: Training history from model.fit()
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy
        axes[0].plot(history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Loss
        axes[1].plot(history.history['loss'], label='Train Loss')
        axes[1].plot(history.history['val_loss'], label='Val Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")


if __name__ == "__main__":
    import tensorflow as tf
    
    # Initialize trainer
    trainer = FakeNewsTrainer(max_words=10000, max_length=500, embedding_dim=100)
    
    # Load and prepare data
    # Download from: https://www.kaggle.com/datasets/emineyetis/fake-news-detection-2024
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.load_and_prepare_data(
        'data/train.csv'
    )
    
    # Train model
    model, history = trainer.train(X_train, X_val, y_train, y_val, epochs=20, batch_size=32)
    
    # Evaluate
    metrics = trainer.evaluate(model, X_test, y_test)
    
    # Save
    trainer.save_model(model)
    trainer.plot_history(history)
    
    logger.info("Training completed!")
