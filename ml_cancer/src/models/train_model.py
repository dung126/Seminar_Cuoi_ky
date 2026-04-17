import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

# Đường dẫn đến dữ liệu đã làm sạch
DATA_PATH = os.path.join('data', 'processed', 'cancer_clean.csv')

# 1. Đọc dữ liệu
df = pd.read_csv(DATA_PATH)
print("== Data info ==")
print(df.info())
print(df.head())
print(df['diagnosis'].unique())

# 2. Tách đặc trưng và nhãn
label_col = 'diagnosis'
X = df.drop(columns=[label_col])
y = df[label_col]

# Kiểm tra nhãn chắc chắn là 0/1
print("Label unique values:", y.unique())

# 3. Chia dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Logistic Regression
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
y_proba_logreg = logreg.predict_proba(X_test)[:, 1]

# 5. Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

# 6. Đánh giá & So sánh
print("\n=== Logistic Regression ===")
print("Accuracy :", accuracy_score(y_test, y_pred_logreg))
print("F1-score :", f1_score(y_test, y_pred_logreg, average='weighted'))
print(classification_report(y_test, y_pred_logreg))
print("\n=== Random Forest ===")
print("Accuracy :", accuracy_score(y_test, y_pred_rf))
print("F1-score :", f1_score(y_test, y_pred_rf, average='weighted'))
print(classification_report(y_test, y_pred_rf))

# 7. Visualization 
os.makedirs('reports/figures', exist_ok=True)

# Confusion Matrix
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred_logreg)).plot(ax=axs[0], cmap="Blues")
axs[0].set_title('Confusion Matrix - Logistic Regression')
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred_rf)).plot(ax=axs[1], cmap="Greens")
axs[1].set_title('Confusion Matrix - Random Forest')
plt.tight_layout()
plt.savefig('reports/figures/confusion_matrices.png')
plt.show()

# ROC Curve
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_logreg)
roc_auc_lr = auc(fpr_lr, tpr_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)
plt.figure(figsize=(8,6))
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC={roc_auc_lr:.2f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={roc_auc_rf:.2f})')
plt.plot([0,1],[0,1],'k--', color='grey')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('reports/figures/roc_curve.png')
plt.show()

# Feature Importance (Random Forest)
importances = rf.feature_importances_
feat_labels = X.columns
indices = importances.argsort()[::-1]
plt.figure(figsize=(10,6))
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), feat_labels[indices], rotation=90)
plt.title("Feature Importance - Random Forest")
plt.tight_layout()
plt.savefig('reports/figures/feature_importance_rf.png')
plt.show()