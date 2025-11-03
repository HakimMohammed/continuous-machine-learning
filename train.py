import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import json
import joblib

# Use centralized hyperparameters
from hyperparams import N_ESTIMATORS, TEST_SIZE, RANDOM_STATE, MODEL_PATH

# Charger les donnees
iris = load_iris()
X, y = iris.data, iris.target

# Split train/test using hyperparameters
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# Entrainement
model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Sauvegarder le modele
joblib.dump(model, 'models/iris_model.pkl')

# Sauvegarder les metriques (include hyperparams for traceability)
metrics = {
    "accuracy": accuracy,
    "n_estimators": N_ESTIMATORS,
    "test_size": TEST_SIZE,
    "random_state": RANDOM_STATE,
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"Accuracy: {accuracy:.4f}")

