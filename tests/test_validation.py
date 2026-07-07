import pandas as pd

from app.model import train_model
from app.validation import (
    validate_schema,
    validate_nulls,
    validate_accuracy,
    validate_precision,
    validate_recall,
    validate_f1
)


df = pd.read_csv("data/breast_cancer.csv")

_, metrics = train_model()


def test_schema():

    assert validate_schema(df)


def test_null_values():

    assert validate_nulls(df)


def test_accuracy_gate():

    assert validate_accuracy(metrics)


def test_precision_gate():

    assert validate_precision(metrics)


def test_recall_gate():

    assert validate_recall(metrics)


def test_f1_gate():

    assert validate_f1(metrics)