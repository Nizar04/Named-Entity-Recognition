import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix
from typing import List, Dict


class MetricsVisualizer:
    def __init__(self, history: Dict = None):
        self.history = history

    def plot_training_history(self, save_path: str = None) -> None:
        fig = make_subplots(rows=2, cols=1,
                            subplot_titles=('Model Accuracy', 'Model Loss'))
        fig.add_trace(
            go.Scatter(y=self.history['accuracy'], name="Training Accuracy"),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(y=self.history['val_accuracy'], name="Validation Accuracy"),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(y=self.history['loss'], name="Training Loss"),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(y=self.history['val_loss'], name="Validation Loss"),
            row=2, col=1
        )
        fig.update_layout(height=600, width=800, title_text="Training Metrics")
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()

    def plot_confusion_matrix(self, y_true: List, y_pred: List,
                              labels: List[str], save_path: str = None) -> None:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.show()

    def plot_entity_distribution(self, tags: List[str], save_path: str = None) -> None:
        tag_counts = pd.Series(tags).value_counts()
        fig = go.Figure(data=[
            go.Bar(x=tag_counts.index, y=tag_counts.values)
        ])
        fig.update_layout(
            title='Distribution of Named Entities',
            xaxis_title='Entity Type',
            yaxis_title='Count',
            height=500,
            width=800
        )
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()

    def plot_sequence_lengths(self, sequences: List[List], save_path: str = None) -> None:
        lengths = [len(seq) for seq in sequences]
        fig = go.Figure(data=[
            go.Histogram(x=lengths, nbinsx=30)
        ])
        fig.update_layout(
            title='Distribution of Sequence Lengths',
            xaxis_title='Sequence Length',
            yaxis_title='Count',
            height=500,
            width=800
        )
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()

    def plot_performance_metrics(self, metrics: Dict[str, float], save_path: str = None) -> None:
        fig = go.Figure(data=[
            go.Bar(x=list(metrics.keys()), y=list(metrics.values()))
        ])
        fig.update_layout(
            title='Model Performance Metrics',
            yaxis_title='Score',
            height=500,
            width=800
        )
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()