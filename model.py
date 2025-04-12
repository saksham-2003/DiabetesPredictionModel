# Importing the required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load the dataset
dataset = pd.read_csv('diabetes_prediction_dataset.csv')

# Encode categorical features
label_encoders = {}
for column in ['gender', 'smoking_history']:
    le = LabelEncoder()
    dataset[column] = le.fit_transform(dataset[column])
    label_encoders[column] = le

# Feature matrix and target vector
features = ["gender", "age", "hypertension", "heart_disease", "smoking_history", 
            "bmi", "HbA1c_level", "blood_glucose_level"]
X = dataset[features]
Y = dataset["diabetes"]

# Scale features using MinMaxScaler instead of StandardScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the data with stratification to preserve class proportions
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=30, stratify=Y)

# Logistic Regression Model with class weighting to handle imbalance
model = LogisticRegression(class_weight='balanced', random_state=30)
model.fit(X_train, Y_train)

# Evaluate the model
Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
report = classification_report(Y_test, Y_pred)

# Optional: Print feature importances (coefficients) for insights
importances = np.abs(model.coef_[0])
sorted_features = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
print("Feature Importances:")
for feat, imp in sorted_features:
    print(f"{feat}: {imp:.4f}")

# Save the model, scaler, and label encoders
pickle.dump(model, open("diabetes_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(label_encoders, open("label_encoders.pkl", "wb"))

# Visualizations
plt.figure(figsize=(15, 8))

# Correlation Heatmap
plt.subplot(2, 2, 1)
sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')

# Age vs. Diabetes Distribution
plt.subplot(2, 2, 2)
sns.histplot(data=dataset, x="age", hue="diabetes", kde=True, bins=30, palette='Set2')
plt.title('Age vs Diabetes')

# BMI vs Diabetes
plt.subplot(2, 2, 3)
sns.boxplot(data=dataset, x="diabetes", y="bmi", palette="pastel")
plt.title('BMI Distribution by Diabetes')

# Smoking History vs Diabetes
plt.subplot(2, 2, 4)
sns.countplot(data=dataset, x="smoking_history", hue="diabetes", palette="viridis")
plt.title('Smoking History by Diabetes')

plt.tight_layout()
plt.show()

# Print results
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", report)
    