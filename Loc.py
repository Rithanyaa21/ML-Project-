import pandas as pd
from geopy.distance import geodesic
import tkinter as tk
from tkinter import messagebox
import folium
import webbrowser

# Step 1: Set the current location (hardcoded for now as Coimbatore Institute of Technology)
def get_user_location():
    latitude = 11.0283  # Latitude for Coimbatore Institute of Technology
    longitude = 77.0270  # Longitude for Coimbatore Institute of Technology
    return latitude, longitude

# Step 2: Load the ambulance dataset
def load_ambulance_data(file_path):
    ambulance_data = pd.read_csv(file_path)
    # Ensure latitude and longitude columns are numerical
    ambulance_data['Location Latitude'] = pd.to_numeric(ambulance_data['Location Latitude'], errors='coerce')
    ambulance_data['Location Longitude'] = pd.to_numeric(ambulance_data['Location Longitude'], errors='coerce')
    return ambulance_data

# Step 3: Find the ambulance location by the selected Location
def find_ambulance_location(ambulance_data, ambulance_location):
    # Filter dataset for the ambulance's arrival location entered by the user
    selected_ambulance = ambulance_data[ambulance_data['Location'].str.lower() == ambulance_location.lower().strip()]
    
    if selected_ambulance.empty:
        raise Exception(f"Ambulance with the specified location '{ambulance_location}' not found in the dataset.")
    
    # Get the latitude and longitude of the ambulance from the dataset
    ambulance_latitude = selected_ambulance.iloc[0]['Location Latitude']
    ambulance_longitude = selected_ambulance.iloc[0]['Location Longitude']
    
    ambulance_position = (ambulance_latitude, ambulance_longitude)
    
    return selected_ambulance.iloc[0], ambulance_position

# Function to calculate the estimated time of arrival based on distance and speed
def calculate_time(distance, speed_kmh):
    # Time = Distance / Speed
    time_hours = distance / speed_kmh
    time_minutes = time_hours * 60  # Convert hours to minutes
    return time_minutes

# Function to retrieve and display the current location, ambulance details, and map
def retrieve_location_and_ambulance():
    try:
        # Get the current user location (Coimbatore Institute of Technology)
        user_location = get_user_location()
        latitude, longitude = user_location
        
        # Load the ambulance data
        file_path = r"C:\Rithanyaa\Machine Learning\ML PROJECT\Corrected_Ambulance_Dataset.csv"
        ambulance_data = load_ambulance_data(file_path)
        
        # Get Ambulance Location from user input
        ambulance_location = ambulance_location_entry.get()
        
        # Find the ambulance by location and calculate distance
        nearest_ambulance, ambulance_position = find_ambulance_location(ambulance_data, ambulance_location)
        
        # Calculate the distance between the current location and the ambulance's location
        distance = geodesic(user_location, ambulance_position).kilometers
        
        # Set an assumed average speed (50 km/h in this example)
        average_speed = 50  # km/h
        
        # Calculate the estimated time to arrival
        estimated_time = calculate_time(distance, average_speed)
        
        # Extract additional information from the nearest ambulance record
        incident_type = nearest_ambulance['Incident Type']
        traffic_condition = nearest_ambulance['Traffic Condition']
        road_condition = nearest_ambulance['Road Condition']
        response_time = nearest_ambulance['Response Time (minutes)']
        
        # Create a map centered at the user's location
        my_map = folium.Map(location=[latitude, longitude], zoom_start=14)
        
        # Add markers for both current location and ambulance location
        folium.Marker([latitude, longitude], popup='Current Location: Coimbatore Institute of Technology').add_to(my_map)
        folium.Marker(ambulance_position, popup=f'Ambulance Location: {ambulance_location}').add_to(my_map)
        
        # Draw a line between the current location and the ambulance's location
        folium.PolyLine([user_location, ambulance_position], color="blue", weight=2.5, opacity=1).add_to(my_map)
        
        # Save the map to an HTML file
        map_file = 'current_and_ambulance_location_map.html'
        my_map.save(map_file)
        
        # Open the map in a web browser
        webbrowser.open(map_file)
        
        # Show information about the ambulance and estimated arrival time
        messagebox.showinfo("Ambulance Details", 
                            f"Ambulance Location: {ambulance_location}\n"
                            f"Distance: {distance:.2f} km\n"
                            f"Incident Type: {incident_type}\n"
                            f"Traffic Condition: {traffic_condition}\n"
                            f"Road Condition: {road_condition}\n"
                            f"Estimated Time of Arrival: {estimated_time:.2f} minutes")
    
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Function to handle button click
def on_location_button_click():
    retrieve_location_and_ambulance()

# Setting up the UI
def setup_ui():
    root = tk.Tk()
    root.title("Ambulance Locator")

    # User Name Input
    tk.Label(root, text="Enter Your Name:").pack(pady=10)
    user_name_entry = tk.Entry(root)
    user_name_entry.pack(pady=10)

    # Ambulance Location Input
    tk.Label(root, text="Enter Ambulance Arrival Location:").pack(pady=10)
    global ambulance_location_entry
    ambulance_location_entry = tk.Entry(root)
    ambulance_location_entry.pack(pady=10)

    # Current Location and Nearest Ambulance Button
    current_location_button = tk.Button(root, text="Get Current Location and Nearest Ambulance", command=on_location_button_click)
    current_location_button.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    setup_ui()
