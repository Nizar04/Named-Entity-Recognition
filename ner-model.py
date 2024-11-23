# model.py

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Conv1D, GlobalMaxPooling1D
import mlflow
import mlflow.tensorflow

class BaseNERModel(ABC):
    def __init__(self, vocab_size: int, n_tags: int, max_len: int):
        self.vocab_size = vocab_size
        self.n_tags = n_tags
        self.max_len = max_len
        self.model: Optional[Sequential] = None
        self.history: Optional[Any] = None
        
    @abstractmethod
    def build_model(self) -> Sequential:
        pass
    
    def save_model(self, path: str) -> None:
        if self.model:
            self.model.save(path)
            
    def load_model(self, path: str) -> None:
        self.model = load_model(path)
    
    def log_metrics(self) -> None:
        if self.history:
            for epoch, metrics in enumerate(self.history.history['accuracy']):
                mlflow.log_metrics({
                    'accuracy': metrics,
                    'val_accuracy': self.history.history['val_accuracy'][epoch],
                    'loss': self.history.history['loss'][epoch],
                    'val_loss': self.history.history['val_loss'][epoch]
                }, step=epoch)

class BiLSTMModel(BaseNERModel):
    def build_model(self) -> Sequential:
        model = Sequential([
            Embedding(self.vocab_size + 1, 128, input_length=self.max_len),
            Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, 
                             recurrent_dropout=0.2)),
            LSTM(64, return_sequences=True, dropout=0.5, recurrent_dropout=0.5),
            TimeDistributed(Dense(self.n_tags, activation="softmax"))
        ])
        
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        self.model = model
        return model

class CNNModel(BaseNERModel):
    def build_model(self) -> Sequential:
        model = Sequential([
            Embedding(self.vocab_size + 1, 128, input_length=self.max_len),
            Conv1D(128, 5, activation='relu'),
            Conv1D(128, 5, activation='relu'),
            TimeDistributed(Dense(self.n_tags, activation="softmax"))
        ])
        
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        self.model = model
        return model

class ModelFactory:
    @staticmethod
    def get_model(model_type: str, vocab_size: int, n_tags: int, max_len: int) -> BaseNERModel:
        models = {
            'bilstm': BiLSTMModel,
            'cnn': CNNModel
        }
        return models[model_type.lower()](vocab_size, n_tags, max_len)
