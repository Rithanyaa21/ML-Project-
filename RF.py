import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, confusion_matrix, classification_report
from sklearn.feature_selection import RFE
from sklearn.tree import plot_tree

# Load the dataset
data_path = r"C:\Rithanyaa\Machine Learning\ML PROJECT\ambulance_dataset.csv"
merged_df = pd.read_csv(data_path)

# Preprocessing function
def preprocess_data(df):
    # Check for missing values
    print("\nMissing values in the dataset:")
    print(df.isnull().sum())
    
    # Drop duplicates
    df = df.drop_duplicates()
    
    # Strip any leading or trailing spaces from object columns
    for col in df.select_dtypes(include=[object]).columns:
        df[col] = df[col].str.strip()
    
    # Normalize column names to remove special characters and spaces
    df.columns = df.columns.str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('Â', '')
    
    return df

# Apply preprocessing
merged_df = preprocess_data(merged_df)

# Print column names for verification
print(merged_df.columns.tolist())

# Define important features and the target variable
important_features = ['Distance_to_Nearest_Hospital_km', 'Temperature_°C', 'Humidity_%', 'Driver_Experience_years']
target = 'Response_Time_minutes'

# Split the data into features and target variable
X = merged_df[important_features]
y = merged_df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Implementing Recursive Feature Elimination (RFE) for feature selection ---

# Initialize a Random Forest Regressor for RFE
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)

# Perform RFE with Random Forest to select the top 3 features
rfe = RFE(estimator=rf_reg, n_features_to_select=3)
rfe.fit(X_train, y_train)

# Get the selected features
selected_features = X_train.columns[rfe.support_]
print(f'Selected features after RFE: {selected_features}')

# Update X_train and X_test with selected features
X_train_rfe = X_train[selected_features]
X_test_rfe = X_test[selected_features]

# --- Train Random Forest with selected features ---

# Train the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_rfe, y_train)

# Make predictions
y_pred = rf_model.predict(X_test_rfe)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f'\nMean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')

# Visualizing the decision tree (first tree in the ensemble)
plt.figure(figsize=(24, 16))  # Increase the figure size for better visibility
plot_tree(rf_model.estimators_[0], feature_names=selected_features, filled=True, max_depth=3)  # Reduced max_depth to 3 for visibility
plt.title('Decision Tree from Random Forest Model (After RFE)', fontsize=18)  # Increase font size for better readability
plt.show()

# EDA Visualizations
plt.figure(figsize=(14, 10))

# Response Time vs. Injury
plt.subplot(2, 2, 1)
sns.boxplot(x='Injury_Severity', y='Response_Time_minutes', data=merged_df)
plt.title('Response Time by Injury Severity')
plt.xticks(rotation=45)

# Distance to Nearest Hospital vs. Injury
plt.subplot(2, 2, 2)
sns.boxplot(x='Injury_Severity', y='Distance_to_Nearest_Hospital_km', data=merged_df)
plt.title('Distance to Nearest Hospital by Injury Severity')
plt.xticks(rotation=45)

# Temperature vs. Response Time
plt.subplot(2, 2, 3)
sns.scatterplot(x='Temperature_°C', y='Response_Time_minutes', data=merged_df)
plt.title('Response Time vs. Temperature')

# Humidity vs. Response Time
plt.subplot(2, 2, 4)
sns.scatterplot(x='Humidity_%', y='Response_Time_minutes', data=merged_df)
plt.title('Response Time vs. Humidity')

plt.tight_layout()
plt.show()

# ---------------------------------------------------
# Random Forest Classification for Patient Survival
# ---------------------------------------------------

# Load another dataset for Patient Survival Prediction
# Assuming you're using the same dataset for patient survival
X_survival = merged_df.drop(columns=['Patient_Survival', 'Response_Time_minutes'])
y_survival = merged_df['Patient_Survival']

# Convert categorical variables to numerical using one-hot encoding (if necessary)
X_survival = pd.get_dummies(X_survival, drop_first=True)

# Split the dataset into training and testing sets
X_train_survival, X_test_survival, y_train_survival, y_test_survival = train_test_split(X_survival, y_survival, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_clf.fit(X_train_survival, y_train_survival)

# Make predictions
y_pred_survival = rf_clf.predict(X_test_survival)

# Evaluate the model
accuracy = accuracy_score(y_test_survival, y_pred_survival)
conf_matrix = confusion_matrix(y_test_survival, y_pred_survival)
class_report = classification_report(y_test_survival, y_pred_survival)

print("\nRandom Forest Classifier - Patient Survival Prediction:")
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
