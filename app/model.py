import joblib
import numpy as np
from app.schemas import PredictionResponse

MODEL_PATH = "models/sentiment_model.joblib"

class SentimentModel:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.label_map = {
            0: "negativo",
            1: "neutral",
            2: "positivo"
        }

    def predict(self, text: str) -> dict:
        probabilities = self.model.predict_proba([text])[0] 
        prediction = np.argmax(probabilities)
        score = float(probabilities[prediction])

        return PredictionResponse(
            text=text,
            sentiment=self.label_map[prediction],
            confidence=round(score,5)
        )