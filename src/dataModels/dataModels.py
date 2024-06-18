from pydantic import BaseModel
from typing import List
import numpy as np

class IrisLabels(BaseModel):
    labels: List[str] = ['setosa', 'versicolor', 'virginica']

    def to_numpy(self) -> np.ndarray:
        return np.array(self.labels)

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class IrisPrediction(BaseModel):
    predictions: List[str]
