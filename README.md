# Advanced Named Entity Recognition (NER) System

A production-ready Named Entity Recognition system built with modern deep learning architectures and best practices.

## Features

- Multiple model architectures (BiLSTM, CNN)
- Advanced data preprocessing options
- MLflow integration for experiment tracking
- FastAPI-based REST API
- Interactive visualizations with Plotly
- Comprehensive metrics and performance analysis
- Model versioning and experiment tracking
- Support for additional linguistic features

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ner-system.git
cd ner-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training a Model

```bash
python main.py \
    --data_path path/to/data.csv \
    --model_type bilstm \
    --batch_size 32 \
    --epochs 10 \
    --use_features \
    --output_dir outputs
```

### Starting the API Server

```bash
python api.py
```

### Making Predictions

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "Apple Inc. CEO Tim Cook announced new iPhone in California"}
)
print(response.json())
```

## Project Structure

```
ner-system/
├── api.py              # FastAPI implementation
├── data_processor.py   # Data processing and feature engineering
├── model.py           # Model architectures and training
├── visualization.py   # Metrics visualization
├── main.py           # Main execution script
├── requirements.txt   # Project dependencies
└── README.md         # This file
```

## Model Architectures

### BiLSTM Model
- Bidirectional LSTM layers
- Dropout for regularization
- TimeDistributed Dense layer for predictions

### CNN Model
- Convolutional layers for feature extraction
- Global max pooling
- Dense layer for predictions

## API Endpoints

### POST /predict
Predicts named entities in the provided text.

Request body:
```json
{
    "text": "Your text here"
}
```

Response:
```json
{
    "entities": [
        {
            "word": "Apple",
            "tag": "ORG"
        },
        ...
    ]
}
```

## Visualization

The system generates several interactive visualizations:
- Training history (accuracy and loss)
- Entity distribution
- Sequence length distribution
- Performance metrics

## Monitoring and Tracking

- MLflow integration for experiment tracking
- Metrics logging
- Model versioning
- Parameter tracking

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Nizar El Mouaquit - nizarelmouaquit@protonmail.com
