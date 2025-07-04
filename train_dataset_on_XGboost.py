# 📦 Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier, plot_importance

# 📥 Load dataset
data = pd.read_csv('D:\\XGBoostClassifier\\diet_recommendations_dataset.csv')

# 🧹 Drop non-informative column
data.drop(['Patient_ID'], axis=1, inplace=True)

# 🔄 Label Encoding for categorical columns
label_encoders = {}
for column in data.columns:
    if data[column].dtype == 'object':
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

# 🎯 Feature/Target split
X = data.drop('Diet_Recommendation', axis=1)
y = data['Diet_Recommendation']

# 🧪 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🚀 Train XGBoost Model
model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# 🔮 Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# ✅ Accuracy
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 🔲 1. Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.tight_layout()
plt.show()

# 📊 2. Classification Metrics Bar Chart
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose().drop(['accuracy', 'macro avg', 'weighted avg'])
report_df[['precision', 'recall', 'f1-score']].plot(kind='bar', figsize=(10, 6))
plt.title('Classification Metrics per Diet Class')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# 🌟 3. XGBoost Feature Importances
plt.figure(figsize=(12, 6))
plot_importance(model, max_num_features=10)
plt.title('Top 10 Feature Importances')
plt.tight_layout()
plt.show()

# 🔮 4. Prediction Confidence Distribution
plt.figure(figsize=(10, 6))
sns.histplot(y_proba.max(axis=1), bins=20, kde=True, color='green')
plt.title('Prediction Confidence Scores')
plt.xlabel('Max Probability per Prediction')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()