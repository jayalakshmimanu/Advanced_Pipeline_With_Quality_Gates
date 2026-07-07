from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_home():

    response = client.get("/")

    assert response.status_code == 200

    assert response.json() == {
        "message": "Advanced Testing Pipeline Working Successfully"
    }


def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json() == {
        "status": "Healthy"
    }


def test_metrics():

    response = client.get("/metrics")

    assert response.status_code == 200

    body = response.json()

    assert "accuracy" in body
    assert "precision" in body
    assert "recall" in body
    assert "f1" in body


def test_predict():

    sample = {
        "features": [
            17.99,
            10.38,
            122.8,
            1001.0,
            0.1184,
            0.2776,
            0.3001,
            0.1471,
            0.2419,
            0.07871,
            1.095,
            0.9053,
            8.589,
            153.4,
            0.006399,
            0.04904,
            0.05373,
            0.01587,
            0.03003,
            0.006193,
            25.38,
            17.33,
            184.6,
            2019.0,
            0.1622,
            0.6656,
            0.7119,
            0.2654,
            0.4601,
            0.1189
        ]
    }

    response = client.post("/predict", json=sample)

    assert response.status_code == 200

    assert response.json()["prediction"] in [
        "Benign",
        "Malignant"
    ]


def test_invalid_prediction():

    sample = {
        "features": [1, 2, 3]
    }

    response = client.post("/predict", json=sample)

    assert response.status_code == 400