import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

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
data.dropna(subset=['Patient Survival'], inplace=True)

# Fill missing values in other columns
data.fillna({
    'Response Time (minutes)': data['Response Time (minutes)'].median(),
    'Injury Severity': data['Injury Severity'].mode()[0],
}, inplace=True)

# Convert categorical columns to 'category' dtype
categorical_columns = data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    data[col] = data[col].astype('category')

# Check the data types after preprocessing
print("\nData Types After Preprocessing:")
print(data.dtypes)

# Exploratory Data Analysis (EDA) remains unchanged
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

# --- Logistic Regression for Patient Survival Prediction ---
# Encoding categorical features with one-hot encoding
X = pd.get_dummies(data.drop(columns=['Patient Survival']), drop_first=True)
y = data['Patient Survival'].astype('category').cat.codes  # Convert to 0 and 1 (Not Survived = 0, Survived = 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Make predictions
y_pred = log_reg.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("\nLogistic Regression - Patient Survival Prediction:")
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# --- Visualizations for Logistic Regression ---

# ROC Curve
y_pred_prob = log_reg.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = roc_auc_score(y_test, y_pred_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
