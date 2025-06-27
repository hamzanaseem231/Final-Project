# --Smart Nutrition App Project--
# Using RandomForest Classifier Model
# Submitted By: Hamza Qasim Mirza (F2021266425)
#               Hamza Naseem (F2021266231)                

import pandas as pd

# Load dataset
data = pd.read_csv('C:/Users/IT/Downloads/diet_recommendations_dataset.csv')

# View first few rows
print(data.head())

# Check column names and data types
print(data.info())

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
data = pd.read_csv('C:/Users/IT/Downloads/diet_recommendations_dataset.csv')

# 2. Check the dataset structure
print("\nDataset Head:\n", data.head())
print("\nDataset Info:\n")
print(data.info())
print("\nMissing values:\n", data.isnull().sum())

# 3. Encode categorical columns (like Gender, Activity_Level, Health_Condition, Recommended_Diet)
label_encoders = {}
for column in data.columns:
    if data[column].dtype == 'object':
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

print("\nEncoded Dataset:\n", data.head())

# 4. Split features and target
X = data.drop('Diet_Recommendation', axis=1)   # Features
y = data['Diet_Recommendation']                # Target

# 5. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Make predictions
y_pred = model.predict(X_test)

# 8. Evaluate model
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 9. Feature importance visualization
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=feature_importances.index)
plt.title('Feature Importances from Random Forest')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()
