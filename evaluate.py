import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    RocCurveDisplay,
)

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Recreate the dataset/split (same as in [train.py](train.py) where [`iris`](train.py) was used)
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load trained model saved by [train.py](train.py) as [`model`](train.py)
model_path = "models/iris_model.pkl"
model = joblib.load(model_path)

# Predictions
y_pred = model.predict(X_test)

# 1) Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
conf_mat_file = os.path.join(OUT_DIR, "confusion_matrix.png")
plt.savefig(conf_mat_file)
plt.close()
print(f"Saved confusion matrix to {conf_mat_file}")

# 2) ROC Curve (binary only)
unique_labels = np.unique(y_test)
if unique_labels.size == 2:
    # binary classification: use predicted probabilities for positive class
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        # fallback to decision function if available
        y_score = model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_file = os.path.join(OUT_DIR, "roc_curve.png")
    plt.savefig(roc_file)
    plt.close()
    print(f"Saved ROC curve to {roc_file}")
else:
    # For multiclass, optionally you could compute one-vs-rest ROC curves.
    print("ROC curve skipped: not a binary classification problem (dataset has "
          f"{unique_labels.size} classes).")

# 3) Distribution of feature importances
if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]
    plt.figure(figsize=(8, 4))
    sns.barplot(x=importances[idx], y=np.array(feature_names)[idx], palette="viridis")
    plt.title("Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    fi_file = os.path.join(OUT_DIR, "feature_importances.png")
    plt.savefig(fi_file)
    plt.close()
    print(f"Saved feature importances to {fi_file}")
else:
    print("Model has no attribute 'feature_importances_'; cannot plot feature importances.")