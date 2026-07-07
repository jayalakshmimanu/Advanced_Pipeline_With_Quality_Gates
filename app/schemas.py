from pydantic import BaseModel


class PredictionInput(BaseModel):
    features: list[float]