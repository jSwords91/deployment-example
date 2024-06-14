from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import logging

from .utils import load_data, split_data, save_model

logging.basicConfig(level=logging.INFO)

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def predict_model(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred

def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def train_and_save_model(model_path: str) -> None:
    # Load dataset
    logging.info("Loading dataset.")
    X, y = load_data()
    logging.info(f"Dataset len: {len(X)}")

    # Split dataset
    logging.info("Splitting dataset.")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.4)

    # Train model
    logging.info("Training Model.")
    model = train_model(X_train, y_train)

    # Evaluate model
    logging.info("Evaluating Model.")
    y_pred = predict_model(model, X_test)
    accuracy = evaluate_model(y_test, y_pred)
    logging.info(f"Model accuracy: {accuracy:.2f}")

    # Save model
    logging.info("Saving Model.")
    save_model(model, model_path)

    logging.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_and_save_model("./src/model/iris_model.pkl")