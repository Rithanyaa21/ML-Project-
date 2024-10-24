import tkinter as tk
from tkinter import messagebox
from pymongo import MongoClient
import pandas as pd
from geopy.distance import geodesic
import folium
import webbrowser
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# MongoDB connection setup
client = MongoClient('mongodb://localhost:27017/')
db = client['ambulance_booking']
user_collection = db['users']
ambulance_collection = db['ambulance_data']  # New collection for storing ambulance data

# Load and preprocess the ambulance dataset
def load_and_preprocess_data(file_path):
    merged_df = pd.read_csv(file_path)
    
    def preprocess_data(df):
        print("\nMissing values in the dataset:")
        print(df.isnull().sum())
        df = df.drop_duplicates()
        for col in df.select_dtypes(include=[object]).columns:
            df[col] = df[col].str.strip()
        return df

    merged_df = preprocess_data(merged_df)
    return merged_df

data_path = r"C:\Rithanyaa\Machine Learning\ML PROJECT\Updated_Ambulance_Dataset.csv"
ambulance_data = load_and_preprocess_data(data_path)

# Train Random Forest model for predicting response time
def train_model():
    important_features = ['Distance to Nearest Hospital (km)', 'Temperature (°C)', 'Humidity (%)', 'Driver Experience (years)']
    target = 'Response Time (minutes)'

    X = ambulance_data[important_features]
    y = ambulance_data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f'\nMean Absolute Error: {mae}')
    print(f'Mean Squared Error: {mse}')
    
    return rf_model

rf_model = train_model()

# Train Logistic Regression model for predicting patient survival
def train_logistic_regression():
    target = 'Patient Survival'
    features = ['Response Time (minutes)', 'Incident Type', 'Traffic Condition', 'Temperature (°C)', 
                'Humidity (%)', 'Location Latitude', 'Location Longitude', 
                'Distance to Nearest Hospital (km)', 'Injury Severity', 'Driver Experience (years)', 
                'Patient Age', 'Heart Rate', 'Blood Pressure']

    X = ambulance_data[features]
    y = ambulance_data[target]

    # Apply get_dummies to one-hot encode categorical features
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    # Store column names to ensure consistency during prediction
    global logistic_regression_columns
    logistic_regression_columns = X_encoded.columns
    
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # Apply class weight balancing
    log_reg_model = LogisticRegression(max_iter=1000, class_weight='balanced')
    log_reg_model.fit(X_train, y_train)

    y_pred = log_reg_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Logistic Regression Model Accuracy: {accuracy * 100:.2f}%')
    print(f'Classification Report:\n{classification_report(y_test, y_pred)}')

    return log_reg_model


log_reg_model = train_logistic_regression()

def get_user_location():
    # Replace with actual user location fetching logic
    latitude = 11.0283
    longitude = 77.0270
    return latitude, longitude

def find_ambulance_location(ambulance_location):
    selected_ambulance = ambulance_data[ambulance_data['Location'].str.lower() == ambulance_location.lower().strip()]
    
    if selected_ambulance.empty:
        raise Exception(f"Ambulance with the specified location '{ambulance_location}' not found in the dataset.")
    
    ambulance_latitude = selected_ambulance.iloc[0]['Location Latitude']
    ambulance_longitude = selected_ambulance.iloc[0]['Location Longitude']
    
    ambulance_position = (ambulance_latitude, ambulance_longitude)
    
    return selected_ambulance.iloc[0], ambulance_position

def calculate_time(distance, traffic_condition):
    # Define realistic speeds in km/h based on traffic conditions
    traffic_speeds = {
        'Light': 60,  # 60 km/h for light traffic
        'Moderate': 40,  # 40 km/h for moderate traffic
        'Heavy': 20  # 20 km/h for heavy traffic
    }
    
    # Default to 'Moderate' traffic speed if condition not found
    speed_kmh = traffic_speeds.get(traffic_condition, 40)
    
    time_hours = distance / speed_kmh  # Time in hours
    time_minutes = time_hours * 60  # Convert to minutes
    return time_minutes


def show_map(user_location, ambulance_position, ambulance_location):
    latitude, longitude = user_location
    
    my_map = folium.Map(location=[latitude, longitude], zoom_start=14)
    folium.Marker([latitude, longitude], popup='Current Location').add_to(my_map)
    folium.Marker(ambulance_position, popup=f'Ambulance Location: {ambulance_location}').add_to(my_map)
    folium.PolyLine([user_location, ambulance_position], color="blue", weight=2.5, opacity=1).add_to(my_map)

    map_file = 'current_and_ambulance_location_map.html'
    my_map.save(map_file)
    webbrowser.open(map_file)

def retrieve_location_and_ambulance(ambulance_location, issue_type):
    try:
        user_location = get_user_location()
        
        nearest_ambulance, ambulance_position = find_ambulance_location(ambulance_location)
        
        distance = geodesic(user_location, ambulance_position).kilometers
        
        # Use the ambulance's traffic condition for more accurate speed calculation
        traffic_condition = nearest_ambulance['Traffic Condition']
        estimated_time = calculate_time(distance, traffic_condition)

        # Input features for response time prediction (without 'Response Time (minutes)')
        input_data = pd.DataFrame({
            'Distance to Nearest Hospital (km)': [distance],
            'Temperature (°C)': [nearest_ambulance['Temperature (°C)']],
            'Humidity (%)': [nearest_ambulance['Humidity (%)']],
            'Driver Experience (years)': [nearest_ambulance['Driver Experience (years)']]
        })
        
        predicted_time = rf_model.predict(input_data)[0]
        
        # Display either the calculated time or the predicted model time
        estimated_time = min(estimated_time, predicted_time)

        # Prepare input for survival prediction
        survival_input = pd.DataFrame({
            'Response Time (minutes)': [predicted_time],  # Use predicted response time
            'Incident Type': [issue_type],  # Use issue type as Incident Type
            'Traffic Condition': [nearest_ambulance['Traffic Condition']],
            'Temperature (°C)': [nearest_ambulance['Temperature (°C)']],
            'Humidity (%)': [nearest_ambulance['Humidity (%)']],
            'Location Latitude': [nearest_ambulance['Location Latitude']],
            'Location Longitude': [nearest_ambulance['Location Longitude']],
            'Distance to Nearest Hospital (km)': [nearest_ambulance['Distance to Nearest Hospital (km)']],
            'Injury Severity': [nearest_ambulance['Injury Severity']],
            'Driver Experience (years)': [nearest_ambulance['Driver Experience (years)']],
            'Patient Age': [nearest_ambulance['Patient Age']],
            'Heart Rate': [nearest_ambulance['Heart Rate']],
            'Blood Pressure': [nearest_ambulance['Blood Pressure']]
        })
        
        # One-hot encode the input data for logistic regression model
        survival_input_encoded = pd.get_dummies(survival_input, drop_first=True)
        
        # Ensure the columns in the prediction data match the training data
        survival_input_encoded = survival_input_encoded.reindex(columns=logistic_regression_columns, fill_value=0)
        
        survival_prediction = log_reg_model.predict(survival_input_encoded)[0]
        survival_status = "Yes" if survival_prediction == 1 else "No"
        
        ambulance_location_display = nearest_ambulance['Location']
        road_condition = nearest_ambulance['Road Condition']
        
        show_map(user_location, ambulance_position, ambulance_location)
        
        messagebox.showinfo("Ambulance Details", 
                            f"Issue Type: {issue_type}\n"
                            f"Ambulance Location: {ambulance_location_display}\n"
                            f"Distance: {distance:.2f} km\n"
                            f"Traffic Condition: {traffic_condition}\n"
                            f"Road Condition: {road_condition}\n"
                            f"Estimated Time of Arrival: {estimated_time:.2f} minutes\n"
                            f"Patient Survival Prediction: {survival_status}")
    
    except Exception as e:
        messagebox.showerror("Error", str(e))


def book_ambulance():
    ambulance_location = ambulance_location_entry.get().strip()
    issue_type = issue_type_entry.get().strip()
    
    if not ambulance_location or not issue_type:
        messagebox.showerror("Input Error", "Please enter both ambulance location and issue type.")
        return

    retrieve_location_and_ambulance(ambulance_location, issue_type)

def signup(name_entry, age_entry, mobile_entry, email_entry, location_entry, password_entry, confirm_password_entry):
    name = name_entry.get().strip()
    age = age_entry.get().strip()
    mobile = mobile_entry.get().strip()
    email = email_entry.get().strip()
    location = location_entry.get().strip()
    password = password_entry.get().strip()
    confirm_password = confirm_password_entry.get().strip()

    # Validate age
    if not age.isdigit() or int(age) < 18:
        messagebox.showerror("Signup Error", "Age must be a number and at least 18.")
        return

    # Validate mobile number
    if not mobile.isdigit() or len(mobile) != 10:
        messagebox.showerror("Signup Error", "Mobile number must be 10 digits.")
        return

    # Validate email
    if '@' not in email or '.' not in email.split('@')[-1]:
        messagebox.showerror("Signup Error", "Please enter a valid email address.")
        return

    # Check if passwords match
    if password != confirm_password:
        messagebox.showerror("Signup Error", "Passwords do not match.")
        return

    # Save user to MongoDB
    user_collection.insert_one({
        "name": name,
        "age": int(age),
        "mobile": mobile,
        "email": email,
        "location": location,
        "password": password
    })

    messagebox.showinfo("Signup Successful", "You have successfully signed up!")

def login(email_entry, password_entry):
    email = email_entry.get().strip()
    password = password_entry.get().strip()

    # Validate login
    user = user_collection.find_one({"email": email, "password": password})
    if user:
        messagebox.showinfo("Login Successful", f"Welcome, {user['name']}!")
        # Show the main booking interface
        main_booking_interface()
    else:
        messagebox.showerror("Login Error", "Invalid email or password.")

def main_booking_interface():
    for widget in root.winfo_children():
        widget.destroy()
    
    welcome_label = tk.Label(root, text="Welcome to the Ambulance Booking System", font=("Arial", 16))
    welcome_label.pack(pady=20)

    global ambulance_location_entry, issue_type_entry
    ambulance_location_entry = tk.Entry(root, width=50)
    ambulance_location_entry.pack(pady=5)
    ambulance_location_entry.insert(0, "Enter Ambulance Location")

    issue_type_entry = tk.Entry(root, width=50)
    issue_type_entry.pack(pady=5)
    issue_type_entry.insert(0, "Enter Issue Type")

    book_button = tk.Button(root, text="Book Ambulance", command=book_ambulance)
    book_button.pack(pady=10)

def signup_interface():
    for widget in root.winfo_children():
        widget.destroy()

    signup_label = tk.Label(root, text="Signup", font=("Arial", 16))
    signup_label.pack(pady=20)

    name_label = tk.Label(root, text="Name:")
    name_label.pack()
    name_entry = tk.Entry(root)
    name_entry.pack()

    age_label = tk.Label(root, text="Age:")
    age_label.pack()
    age_entry = tk.Entry(root)
    age_entry.pack()

    mobile_label = tk.Label(root, text="Mobile Number:")
    mobile_label.pack()
    mobile_entry = tk.Entry(root)
    mobile_entry.pack()

    email_label = tk.Label(root, text="Email:")
    email_label.pack()
    email_entry = tk.Entry(root)
    email_entry.pack()

    location_label = tk.Label(root, text="Location:")
    location_label.pack()
    location_entry = tk.Entry(root)
    location_entry.pack()

    password_label = tk.Label(root, text="Password:")
    password_label.pack()
    password_entry = tk.Entry(root, show="*")
    password_entry.pack()

    confirm_password_label = tk.Label(root, text="Confirm Password:")
    confirm_password_label.pack()
    confirm_password_entry = tk.Entry(root, show="*")
    confirm_password_entry.pack()

    signup_button = tk.Button(root, text="Sign Up", command=lambda: signup(name_entry, age_entry, mobile_entry, email_entry, location_entry, password_entry, confirm_password_entry))
    signup_button.pack(pady=10)

    back_button = tk.Button(root, text="Back to Login", command=login_interface)
    back_button.pack()

def login_interface():
    for widget in root.winfo_children():
        widget.destroy()

    login_label = tk.Label(root, text="Login", font=("Arial", 16))
    login_label.pack(pady=20)

    email_label = tk.Label(root, text="Email:")
    email_label.pack()
    email_entry = tk.Entry(root)
    email_entry.pack()

    password_label = tk.Label(root, text="Password:")
    password_label.pack()
    password_entry = tk.Entry(root, show="*")
    password_entry.pack()

    login_button = tk.Button(root, text="Login", command=lambda: login(email_entry, password_entry))
    login_button.pack(pady=10)

    signup_button = tk.Button(root, text="Sign Up", command=signup_interface)
    signup_button.pack()

root = tk.Tk()
root.title("Ambulance Booking System")
root.geometry("600x400")

# Start with the login interface
login_interface()

root.mainloop()
