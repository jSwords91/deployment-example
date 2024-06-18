import azure.functions as func
import logging
import joblib
import numpy as np
from src.dataModels.dataModels import IrisLabels, IrisFeatures, IrisPrediction

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

logging.basicConfig(level=logging.INFO)

# Load model and class labels at startup
model = joblib.load("src/model/iris_model.pkl")
logging.info("Model load successful.")

class_labels = IrisLabels().to_numpy()
logging.info("Class labels loaded.")

@app.function_name(name="PredictFunction")
@app.route(route="predict", methods=["POST"])
def predict(req: func.HttpRequest) -> func.HttpResponse:
    try:
        # Parse request JSON body

        body = req.get_json()
        features = [IrisFeatures(**item) for item in body]
        logging.info(f"Received features: {features}")

        # Prepare input data for prediction
        input_data = np.array([[f.sepal_length, f.sepal_width, f.petal_length, f.petal_width] for f in features])
        logging.info(f"Input data: {input_data}")

        # Predict using the loaded model
        predictions = model.predict(input_data)
        predicted_classes = [class_labels[pred] for pred in predictions]

        # Return the predictions as JSON
        return func.HttpResponse(
            body=IrisPrediction(predictions=predicted_classes).json(),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        return func.HttpResponse(
            body=str(e),
            status_code=400
        )

@app.function_name(name="home")
@app.route(route="", methods=["GET"])
def read_root(req: func.HttpRequest) -> func.HttpResponse:
    return func.HttpResponse(
        body="""
                Welcome to the Iris Prediction API!

                Usage Instructions:
                -------------------
                This API allows you to predict the species of an iris flower based on the length and width of its sepals and petals.

                Method: POST
                URL: https://myirispredictionfunction.azurewebsites.net/api/predict

                Example Request:
                ----------------
                URL: https://myirispredictionfunction.azurewebsites.net/api/predict
                Method: POST
                Headers: {
                "Content-Type": "application/json"
                }
                Body: [
                {
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2
                }
                ]

                Example Code (Python):
                -----------------------
                import requests

                url = "https://myirispredictionfunction.azurewebsites.net/api/predict"
                data = [
                    {
                        "sepal_length": 5.1,
                        "sepal_width": 3.5,
                        "petal_length": 1.4,
                        "petal_width": 0.2
                    }
                ]
                response = requests.post(url, json=data)
                prediction = response.json()
                print(f"Predicted flower species: {prediction['predictions'][0]}")
                """,
        status_code=200,
        mimetype="text/plain"
    )