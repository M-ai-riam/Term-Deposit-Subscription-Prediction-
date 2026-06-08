# %%
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import shap 
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score,  roc_curve, auc, roc_auc_score, ConfusionMatrixDisplay 

# %%
df= pd.read_csv("bank_additional_full.csv",sep=',')

# %%
df.dtypes

# %%
df.shape

# %%
df.describe()

# %%
df.isnull().sum()

# %%
df

# %%
df_processed = df.copy()
categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()


# %%
le_target = LabelEncoder()
df_processed['y'] = le_target.fit_transform(df_processed['y'])

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col])
    label_encoders[col] = le

# %%
X = df_processed.drop('y', axis=1)
y = df_processed['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# %%
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)



# %%

rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)

# %%
y_pred_lr = lr_model.predict(X_test_scaled)
y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
y_pred_rf = rf_model.predict(X_test_scaled)
y_pred_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

lr_accuracy = accuracy_score(y_test, y_pred_lr)
lr_f1 = f1_score(y_test, y_pred_lr)
lr_auc = roc_auc_score(y_test, y_pred_proba_lr)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)
rf_auc = roc_auc_score(y_test, y_pred_proba_rf)
print(f"\nLogistic Regression:\n  Accuracy: {lr_accuracy}, F1: {lr_f1}, AUC: {lr_auc}")
print(f"\nRandom Forest:\n  Accuracy: {rf_accuracy}, F1: {rf_f1}, AUC: {rf_auc}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

cm_lr = confusion_matrix(y_test, y_pred_lr)
disp_lr = ConfusionMatrixDisplay(confusion_matrix=cm_lr, display_labels=['No', 'Yes'])
disp_lr.plot(ax=axes[0], cmap='Blues', values_format='d')
axes[0].set_title('Logistic Regression')

cm_rf = confusion_matrix(y_test, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=['No', 'Yes'])
disp_rf.plot(ax=axes[1], cmap='Greens', values_format='d')
axes[1].set_title('Random Forest')

plt.tight_layout()
plt.show()

# %%
fig, ax = plt.subplots(figsize=(8, 6))
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)

ax.plot(fpr_lr, tpr_lr, label=f'LR (AUC={lr_auc:.3f})', linewidth=2.5, color='#FF6B6B')
ax.plot(fpr_rf, tpr_rf, label=f'RF (AUC={rf_auc:.3f})', linewidth=2.5, color='#4ECDC4')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# SHAP explainability
X_test_sample = X_test_scaled.iloc[:100]

explainer_lr = shap.LinearExplainer(lr_model, X_test_scaled)
shap_values_lr = explainer_lr.shap_values(X_test_sample)

explainer_rf = shap.TreeExplainer(rf_model)
shap_values_rf = explainer_rf.shap_values(X_test_sample)
if isinstance(shap_values_rf, list):
    shap_values_rf = shap_values_rf[1]


# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plt.subplot(1, 2, 1)
shap.summary_plot(shap_values_lr, X_test_sample, plot_type="bar", show=False)
plt.title('Logistic Regression - Feature Importance')
plt.subplot(1, 2, 2)
shap.summary_plot(shap_values_rf, X_test_sample, plot_type="bar", show=False)
plt.title('\n Random Forest - Feature Importance')
plt.tight_layout()
plt.show()



for idx in range(5):
    actual = y_test.iloc[idx]
    print(f"\nPrediction #{idx + 1}:")
    print(f"  Actual: {'YES' if actual == 1 else 'NO'}")
    print(f"  LR: {'YES' if y_pred_lr[idx] == 1 else 'NO'} ({y_pred_proba_lr[idx]:.2%})")
    print(f"  RF: {'YES' if y_pred_rf[idx] == 1 else 'NO'} ({y_pred_proba_rf[idx]:.2%})")




%



