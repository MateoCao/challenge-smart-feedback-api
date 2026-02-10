from pydantic import BaseModel

class PredictionRequest(BaseModel):
    text: str
    
class PredictionResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float