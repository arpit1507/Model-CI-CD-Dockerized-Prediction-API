import os
import json
from src.train import train_and_save
from src.model import predict, load_model
from sklearn.datasets import load_breast_cancer

def test_training_and_model_file_created(tmp_path):
    # train and save model (uses models/ by default)
    res = train_and_save(random_state=0)
    assert os.path.exists(res["model_path"])

def test_predict_output_shape():
    data, _ = load_breast_cancer(return_X_y=True)
    sample = data[:2].tolist()
    preds = predict(sample)
    assert isinstance(preds, list)
    assert len(preds) == 2
    # each probability must be between 0 and 1
    assert all(0.0 <= p <= 1.0 for p in preds)