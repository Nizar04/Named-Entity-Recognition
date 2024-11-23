# main.py

import os
import argparse
import mlflow
from model import ModelFactory
from data_processor import NERDataProcessor
from visualization import MetricsVisualizer
from sklearn.metrics import classification_report
import json

def train_model(args):
    # Initialize MLflow
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment_name)
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            "model_type": args.model_type,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "use_features": args.use_features
        })
        
        # Process data
        processor = NERDataProcessor(args.data_path, use_additional_features=args.use_features)
        train_data, test_data, train_tags, test_tags = processor.prepare_sequences()
        
        # Initialize and train model
        model = ModelFactory.get_model(
            args.model_type,
            vocab_size=len(processor.token2idx),
            n_tags=processor.n_tags,
            max_len=processor.max_len
        )
        
        model.build_model()
        history = model.train(
            train_data,
            train_tags,
            validation_split=0.2,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        # Save model
        model.save_model(args.model_path)
        
        # Visualize results
        visualizer = MetricsVisualizer(history.history)
        visualizer.plot_training_history(os.path.join(args.output_dir, 'training_history.html'))
        
        # Generate predictions and metrics
        predictions = model.predict(test_data)
        metrics = classification_report(
            test_tags.argmax(axis=-1).flatten(),
            predictions.argmax(axis=-1).flatten(),
            target_names=list(processor.tag2idx.keys()),
            output_dict=True
        )
        
        # Log metrics
        for label, m in metrics.items():
            if isinstance(m, dict):
                for metric_name, value in m.items():
                    mlflow.log_metric(f"{label}_{metric_name}", value)
        
        # Save metrics visualization
        visualizer.plot_performance_metrics(
            {k: v['f1-score'] for k, v in metrics.items() if isinstance(v, dict)},
            os.path.join(args.output_dir, 'performance_metrics.html')
        )
        
        # Save entity distribution
        visualizer.plot_entity_distribution(
            [processor.idx2tag[idx] for idx in test_tags.argmax(axis=-1).flatten()],
            os.path.join(args.output_dir, 'entity_distribution.html')
        )
        
        return model, processor

def main():
    parser = argparse.ArgumentParser(description='Train NER model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to training data')
    parser.add_argument('--model_path', type=str, default='model.h5', help='Path to save model')
    parser.add_argument('--model_type', type=str, default='bilstm', choices=['bilstm', 'cnn'],
                      help='Type of model to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--use_features', action='store_true',
                      help='Whether to use additional features')
    parser.add_argument('--output_dir', type=str, default='outputs',
                      help='Directory to save outputs')
    parser.add_argument('--mlflow_uri', type=str, default='http://localhost:5000',
                      help='MLflow tracking URI')
    parser.add_argument('--experiment_name', type=str, default='ner_training',
                      help='MLflow experiment name')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train model
    model, processor = train_model(args)
    
    # Save processor vocabulary
    with open(os.path.join(args.output_dir, 'vocab.json'), 'w') as f:
        json.dump({
            'token2idx': processor.token2idx,
            'tag2idx': processor.tag2idx
        }, f)

if __name__ == "__main__":
    main()
