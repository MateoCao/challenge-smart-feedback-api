from pydantic import BaseModel
from typing import List

class PredictionRequest(BaseModel):
    texts: List[str]