from joblib import load
import os
MODEL_PATH = os.path.join("models", "model.pkl")

_model = None

def load_model(path: str = MODEL_PATH):
    global _model
    if _model is None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found at {path}. Run `python src/train.py` first.")
        _model = load(path)
    return _model

def predict(inputs):
    """
    inputs: list of list of floats (n_samples x n_features)
    returns: list of probabilities for class 1
    """
    model = load_model()
    probs = model.predict_proba(inputs)[:, 1].tolist()
    return probs