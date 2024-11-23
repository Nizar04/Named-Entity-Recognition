# data_processor.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import spacy

class DataPreprocessor:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        
    def clean_text(self, text: str) -> str:
        doc = self.nlp(text)
        tokens = [token.text for token in doc if not token.is_space]
        return " ".join(tokens)
    
    def add_pos_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['POS'] = df['Word'].apply(lambda x: self.nlp(str(x))[0].pos_)
        return df
    
    def add_dependency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['DEP'] = df['Word'].apply(lambda x: self.nlp(str(x))[0].dep_)
        return df

class NERDataProcessor:
    def __init__(self, data_path: str, use_additional_features: bool = False):
        self.preprocessor = DataPreprocessor()
        self.data = pd.read_csv(data_path, encoding='unicode_escape')
        self.use_additional_features = use_additional_features
        self.token2idx: Dict = {}
        self.idx2token: Dict = {}
        self.tag2idx: Dict = {}
        self.idx2tag: Dict = {}
        self.max_len: int = 0
        self.n_tags: int = 0
        
        if use_additional_features:
            self.data = self._add_features()
            
        self.setup_mappings()
    
    def _add_features(self) -> pd.DataFrame:
        df = self.preprocessor.add_pos_features(self.data)
        df = self.preprocessor.add_dependency_features(df)
        return df
    
    def setup_mappings(self) -> None:
        words = set(self.data['Word'].dropna())
        tags = set(self.data['Tag'].dropna())
        
        self.token2idx = {word: idx for idx, word in enumerate(words)}
        self.idx2token = {idx: word for word, idx in self.token2idx.items()}
        self.tag2idx = {tag: idx for idx, tag in enumerate(tags)}
        self.idx2tag = {idx: tag for tag, idx in self.tag2idx.items()}
    
    def prepare_sequences(self) -> Tuple:
        self.data['Word_idx'] = self.data['Word'].map(self.token2idx)
        self.data['Tag_idx'] = self.data['Tag'].map(self.tag2idx)
        
        grouped = self.data.fillna(method='ffill').groupby('Sentence #', as_index=False)
        
        if self.use_additional_features:
            grouped = grouped['Word_idx', 'Tag_idx', 'POS', 'DEP'].agg(lambda x: list(x))
            return self._prepare_with_features(grouped)
        else:
            grouped = grouped['Word_idx', 'Tag_idx'].agg(lambda x: list(x))
            return self._prepare_basic(grouped)
    
    def _prepare_basic(self, grouped: pd.DataFrame) -> Tuple:
        tokens = grouped['Word_idx'].tolist()
        tags = grouped['Tag_idx'].tolist()
        
        self.max_len = max(len(s) for s in tokens)
        self.n_tags = len(self.tag2idx)
        
        return self.pad_and_split_data(tokens, tags)
    
    def _prepare_with_features(self, grouped: pd.DataFrame) -> Tuple:
        tokens = grouped['Word_idx'].tolist()
        tags = grouped['Tag_idx'].tolist()
        pos = grouped['POS'].tolist()
        dep = grouped['DEP'].tolist()
        
        self.max_len = max(len(s) for s in tokens)
        self.n_tags = len(self.tag2idx)
        
        return self.pad_and_split_data_with_features(tokens, tags, pos, dep)
    
    def pad_and_split_data(self, tokens: List, tags: List) -> Tuple:
        padded_tokens = pad_sequences(tokens, maxlen=self.max_len, padding='post',
                                    value=len(self.token2idx) - 1)
        padded_tags = pad_sequences(tags, maxlen=self.max_len, padding='post',
                                  value=self.tag2idx['O'])
        padded_tags = [to_categorical(i, num_classes=self.n_tags) for i in padded_tags]
        
        return train_test_split(padded_tokens, padded_tags, test_size=0.2, random_state=42)
    
    def pad_and_split_data_with_features(self, tokens: List, tags: List, 
                                       pos: List, dep: List) -> Tuple:
        # Convert POS and DEP to numerical features
        pos_encoder = LabelEncoder()
        dep_encoder = LabelEncoder()
        
        flat_pos = [item for sublist in pos for item in sublist]
        flat_dep = [item for sublist in dep for item in sublist]
        
        pos_encoded = pos_encoder.fit_transform(flat_pos)
        dep_encoded = dep_encoder.fit_transform(flat_dep)
        
        # Reshape back to original structure
        pos_sequences = [pos_encoded[i:i+len(p)] for i, p in enumerate(pos)]
        dep_sequences = [dep_encoded[i:i+len(d)] for i, d in enumerate(dep)]
        
        # Pad sequences
        padded_tokens = pad_sequences(tokens, maxlen=self.max_len, padding='post',
                                    value=len(self.token2idx) - 1)
        padded_tags = pad_sequences(tags, maxlen=self.max_len, padding='post',
                                  value=self.tag2idx['O'])
        padded_pos = pad_sequences(pos_sequences, maxlen=self.max_len, padding='post',
                                 value=len(pos_encoder.classes_) - 1)
        padded_dep = pad_sequences(dep_sequences, maxlen=self.max_len, padding='post',
                                 value=len(dep_encoder.classes_) - 1)
        
        padded_tags = [to_categorical(i, num_classes=self.n_tags) for i in padded_tags]
        
        # Combine features
        features = np.concatenate([
            padded_tokens.reshape(-1, self.max_len, 1),
            padded_pos.reshape(-1, self.max_len, 1),
            padded_dep.reshape(-1, self.max_len, 1)
        ], axis=-1)
        
        return train_test_split(features, padded_tags, test_size=0.2, random_state=42)
