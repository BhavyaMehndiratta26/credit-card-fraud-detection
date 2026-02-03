# ============================================================
# Credit Card Fraud Detection
# Logistic Regression | Random Forest | Gradient Boosting
# ============================================================

# ------------------------------------------------------------
# 1. Import Required Libraries
# ------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 2. Load Dataset
# ------------------------------------------------------------
df = pd.read_csv("creditcard.csv")
print("Dataset shape:", df.shape)

# ------------------------------------------------------------
# 3. Data Preprocessing
# ------------------------------------------------------------

# Scale Amount feature
scaler = StandardScaler()
df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])

# Drop original Amount and Time columns
df.drop(['Amount', 'Time'], axis=1, inplace=True)

# ------------------------------------------------------------
# 4. Define Features and Target
# ------------------------------------------------------------
X = df.drop('Class', axis=1)
y = df['Class']

# ------------------------------------------------------------
# 5. Train-Test Split (Stratified)
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Fraud cases in training set:", y_train.sum())
print("Fraud cases in test set:", y_test.sum())

# ------------------------------------------------------------
# 6. Model Evaluation Function
# ------------------------------------------------------------
def evaluate_model(name, model):
    print(f"\n{name}")
    print("=" * 60)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Classification Report
    print(classification_report(y_test, y_pred))

    # Confusion Matrix (Print)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Confusion Matrix (Plot)
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, cmap="Blues"
    )
    plt.title(f"{name} - Confusion Matrix")
    plt.show()

    # ROC Curve values
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc

# ------------------------------------------------------------
# 7. Logistic Regression
# ------------------------------------------------------------
lr_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

fpr_lr, tpr_lr, auc_lr = evaluate_model(
    "Logistic Regression", lr_model
)

# ------------------------------------------------------------
# 8. Random Forest
# ------------------------------------------------------------
rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

fpr_rf, tpr_rf, auc_rf = evaluate_model(
    "Random Forest", rf_model
)

# ------------------------------------------------------------
# 9. Gradient Boosting Classifier
# ------------------------------------------------------------
gb_model = GradientBoostingClassifier(
    random_state=42
)

fpr_gb, tpr_gb, auc_gb = evaluate_model(
    "Gradient Boosting", gb_model
)

# ------------------------------------------------------------
# 10. ROC Curve Comparison
# ------------------------------------------------------------
plt.figure(figsize=(7, 6))
plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {auc_lr:.3f})")
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc_rf:.3f})")
plt.plot(fpr_gb, tpr_gb, label=f"Gradient Boosting (AUC = {auc_gb:.3f})")
plt.plot([0, 1], [0, 1], "k--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# ------------------------------------------------------------
# 11. Model Comparison Table
# ------------------------------------------------------------
results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "Gradient Boosting"
    ],
    "ROC-AUC": [
        auc_lr,
        auc_rf,
        auc_gb
    ]
})

print("\nModel Comparison:")
print(results)
