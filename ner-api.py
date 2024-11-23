# api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
from model import ModelFactory
from data_processor import NERDataProcessor, DataPreprocessor

app = FastAPI(title="NER API", description="API for Named Entity Recognition")

class TextInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    entities: List[Dict[str, str]]

class NERService:
    def __init__(self, model_path: str, processor: NERDataProcessor):
        self.model = ModelFactory.get_model('bilstm', 
                                          processor.vocab_size,
                                          processor.n_tags,
                                          processor.max_len)
        self.model.load_model(model_path)
        self.processor = processor
        self.preprocessor = DataPreprocessor()
    
    def predict(self, text: str) -> List[Dict[str, str]]:
        cleaned_text = self.preprocessor.clean_text(text)
        words = cleaned_text.split()
        word_indices = [self.processor.token2idx.get(word, len(self.processor.token2idx)-1) 
                       for word in words]
        
        padded_sequence = pad_sequences([word_indices], 
                                      maxlen=self.processor.max_len,
                                      padding='post',
                                      value=len(self.processor.token2idx)-1)
        
        predictions = self.model.predict(padded_sequence)
        predicted_tags = np.argmax(predictions[0], axis=-1)
        
        entities = []
        for word, tag_idx in zip(words, predicted_tags[:len(words)]):
            entities.append({
                'word': word,
                'tag': self.processor.idx2tag[tag_idx]
            })
        
        return entities

service = NERService('model.h5', NERDataProcessor('ner_dataset.csv'))

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: TextInput) -> PredictionOutput:
    try:
        entities = service.predict(input_data.text)
        return PredictionOutput(entities=entities)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
