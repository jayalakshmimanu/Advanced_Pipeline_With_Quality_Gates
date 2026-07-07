from fastapi import FastAPI, HTTPException
import pandas as pd

from app.schemas import PredictionInput
from app.model import train_model, load_model

app = FastAPI(
    title="Advanced Testing Pipeline",
    version="1.0"
)

# Train model when API starts
model, metrics = train_model()

# Load feature names from dataset
feature_names = pd.read_csv("data/breast_cancer.csv").drop(
    columns=["diagnosis"]
).columns.tolist()


@app.get("/")
def home():
    return {
        "message": "Advanced Testing Pipeline Working Successfully"
    }


@app.get("/health")
def health():
    return {
        "status": "Healthy"
    }


@app.get("/metrics")
def get_metrics():
    return metrics


@app.post("/predict")
def predict(data: PredictionInput):

    if len(data.features) != len(feature_names):
        raise HTTPException(
            status_code=400,
            detail=f"Exactly {len(feature_names)} features are required."
        )

    model = load_model()

    # Create DataFrame with correct feature names
    input_df = pd.DataFrame(
        [data.features],
        columns=feature_names
    )

    prediction = model.predict(input_df)[0]

    result = "Malignant" if prediction == 0 else "Benign"

    return {
        "prediction": result
    }