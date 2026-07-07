import pandas as pd


def validate_schema(df):

    expected_columns = [
        'mean radius',
        'mean texture',
        'mean perimeter',
        'mean area',
        'mean smoothness',
        'mean compactness',
        'mean concavity',
        'mean concave points',
        'mean symmetry',
        'mean fractal dimension',
        'radius error',
        'texture error',
        'perimeter error',
        'area error',
        'smoothness error',
        'compactness error',
        'concavity error',
        'concave points error',
        'symmetry error',
        'fractal dimension error',
        'worst radius',
        'worst texture',
        'worst perimeter',
        'worst area',
        'worst smoothness',
        'worst compactness',
        'worst concavity',
        'worst concave points',
        'worst symmetry',
        'worst fractal dimension',
        'diagnosis'
    ]

    return list(df.columns) == expected_columns


def validate_nulls(df):
    return df.isnull().sum().sum() == 0


def validate_accuracy(metrics, threshold=0.90):
    return metrics["accuracy"] >= threshold


def validate_precision(metrics, threshold=0.90):
    return metrics["precision"] >= threshold


def validate_recall(metrics, threshold=0.90):
    return metrics["recall"] >= threshold


def validate_f1(metrics, threshold=0.90):
    return metrics["f1"] >= threshold