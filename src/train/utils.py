import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from typing import Tuple

def load_data() -> Tuple:
    iris = load_iris()
    return iris.data, iris.target

def split_data(X, y, test_size: float = 0.2, random_state: int = 42) -> Tuple:
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def save_model(model, model_path: str) -> None:
    joblib.dump(model, model_path)
    return None