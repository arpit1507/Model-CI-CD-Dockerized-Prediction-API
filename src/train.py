from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from joblib import dump
import os

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

def train_model():
    X,y=load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    os.makedirs(MODEL_DIR, exist_ok=True)
    dump(clf, MODEL_PATH)
    return {"model_path": MODEL_PATH, "auc": float(accuracy)}

if __init__=="__main__":
    result = train_model()
    print(f"Model saved to {result['model_path']} with accuracy {result['auc']:.4f}")


