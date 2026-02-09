import joblib
import numpy as np

MODEL_PATH = "models/sentiment_model.joblib"

class SentimentModel:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.label_map = {
            0: "negativo",
            1: "neutral",
            2: "positivo"
        }

    def predict(self, texts: list[str]) -> list[dict]:
        probabilities = self.model.predict_proba(texts)
        predictions = np.argmax(probabilities, axis=1)

        results = []
        for pred, probs in zip(predictions, probabilities):
            results.append({
                "sentiment": self.label_map[pred],
                "score": float(np.max(probs))
            })

        return results