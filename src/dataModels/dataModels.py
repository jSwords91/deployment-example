from dataclasses import dataclass
import numpy as np
from typing import List

@dataclass(frozen=True)
class IrisLabels:
    labels: List[str] = ('setosa', 'versicolor', 'virginica')

    def to_numpy(self) -> np.ndarray:
        return np.array(self.labels)

@dataclass
class IrisFeatures:
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@dataclass
class IrisPrediction:
    predictions: List[str]
