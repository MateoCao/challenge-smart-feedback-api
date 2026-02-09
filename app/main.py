from fastapi import FastAPI, HTTPException
from app.model import SentimentModel
from app.schemas  import PredictionRequest

app = FastAPI()
model = SentimentModel()

@app.get("/health")
def health_check():
    return {
        "status": "online"
    }

@app.post("/predict")
def predict_sentiment(request: PredictionRequest):
    try:
        if not request.texts:
            raise HTTPException(status_code=400, detail="La lista de textos no puede estar vacía.")
        return model.predict(request.texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))