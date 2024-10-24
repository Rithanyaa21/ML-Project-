import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
from sklearn.metrics import roc_curve, auc

# Load the dataset using the specified file path
file_path = r"C:\Rithanyaa\Machine Learning\ML PROJECT\ambulance_dataset.csv"
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("Dataset Preview:")
print(data.head())

# Display the column names
print("\nColumn Names:")
print(data.columns.tolist())

# Basic statistics of the dataset
print("\nBasic Statistics:")
print(data.describe(include='all'))

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Drop rows with missing target variable (if any)
# Assuming the target variable is 'Patient Survival'
data.dropna(subset=['Patient Survival'], inplace=True)

# Fill missing values in other columns (example strategies)
data.fillna({
    'Response Time (minutes)': data['Response Time (minutes)'].median(),
    'Injury Severity': data['Injury Severity'].mode()[0],
    # Add more columns as necessary
}, inplace=True)

# Convert categorical columns to 'category' dtype
categorical_columns = data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    data[col] = data[col].astype('category')

# Check the data types after preprocessing
print("\nData Types After Preprocessing:")
print(data.dtypes)

# Exploratory Data Analysis (EDA)

# Count plot for Incident Type vs Patient Survival
plt.figure(figsize=(12, 6))
sns.countplot(x='Incident Type', hue='Patient Survival', data=data)
plt.title('Incident Type vs Patient Survival')
plt.xticks(rotation=45)
plt.legend(title='Patient Survival', loc='upper right', labels=['Not Survived', 'Survived'])
plt.show()

# Box plot for Response Time vs Injury Severity
plt.figure(figsize=(12, 6))
sns.boxplot(x='Injury Severity', y='Response Time (minutes)', data=data)
plt.title('Response Time vs Injury Severity')
plt.show()

# Pairplot to visualize relationships between numerical features
numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()

plt.figure(figsize=(10, 8))
sns.pairplot(data, hue='Patient Survival', vars=numerical_features)
plt.suptitle('Pairplot of Numerical Features', y=1.02)
plt.show()

# Correlation heatmap (only numerical features)
numerical_data = data[numerical_features]
plt.figure(figsize=(14, 8))
sns.heatmap(numerical_data.corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Heatmap')
plt.show()

# Distribution of Response Time
plt.figure(figsize=(12, 6))
sns.histplot(data['Response Time (minutes)'], bins=30, kde=True)
plt.title('Distribution of Response Time (minutes)')
plt.xlabel('Response Time (minutes)')
plt.ylabel('Frequency')
plt.show()

# Additional analysis based on categorical variables (e.g., Patient Survival)
plt.figure(figsize=(12, 6))
sns.countplot(x='Injury Severity', hue='Patient Survival', data=data)
plt.title('Injury Severity vs Patient Survival')
plt.legend(title='Patient Survival', loc='upper right', labels=['Not Survived', 'Survived'])
plt.show()

# XGBoost Model Implementation
# Convert categorical columns to one-hot encoded for XGBoost
data = pd.get_dummies(data, drop_first=True)

# Define the feature set (X) and the target variable (y)
X = data.drop('Patient Survival', axis=1)
y = data['Patient Survival']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the XGBoost classifier
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = xgb_model.predict(X_test_scaled)

# Evaluate the model
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Visualize the confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# ROC Curve and AUC Score
y_pred_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'XGBoost (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.50)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
