from typing import List
import logging
import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from src.dataModels.dataModels import IrisLabels, IrisFeatures, IrisPrediction

app = FastAPI(description="Sample FastAPI Deployment")

logging.basicConfig(level=logging.INFO)

@app.get("/", response_class=HTMLResponse, status_code=200)
def root():
    html_content = """
    <!DOCTYPE html>
    <html>
    <body>
        <h1>Sample Deployment</h1>
        <p>This is a basic HTML response from a FastAPI endpoint.</p>
    </body>
    </html>
    """
    return html_content

logging.info("Loading class labels.")
class_labels = IrisLabels().to_numpy()

@app.on_event(event_type="startup")
async def load_model():
    global model
    model = joblib.load("./src/model/iris_model.pkl")
    logging.info("Model load successful.")


@app.post("/predict", response_model=IrisPrediction)
async def predict(features: List[IrisFeatures]):
    """
    Predict the class of iris flowers.

    Args:
        features (List[IrisFeatures]): A list of IrisFeatures objects.

    Returns:
        dict: A dictionary with the list of predicted class names.

    Example:
        Input: [{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}]
        Output: {"predictions": ["setosa"]}
    """

    input_data = np.array(
        [
            [f.sepal_length, f.sepal_width, f.petal_length, f.petal_width]
            for f in features
        ]
    )
    predictions = model.predict(input_data)
    predicted_classes = [class_labels[pred] for pred in predictions]
    return {"predictions": predicted_classes}
