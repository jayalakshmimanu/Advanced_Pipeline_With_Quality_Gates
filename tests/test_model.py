from app.model import train_model


def test_model_training():

    model, metrics = train_model()

    assert model is not None


def test_accuracy():

    _, metrics = train_model()

    assert metrics["accuracy"] >= 0.90


def test_precision():

    _, metrics = train_model()

    assert metrics["precision"] >= 0.90


def test_recall():

    _, metrics = train_model()

    assert metrics["recall"] >= 0.90


def test_f1_score():

    _, metrics = train_model()

    assert metrics["f1"] >= 0.90