""""
Program name: Evaluation.py
----------------------------------------------------------------------
Program elaborated to evaluate data from a MySQL database and insert spark events into a second table in the database.
----------------------------------------------------------------------
Description:
Python program to evaluate waveform data from a MySQL database using a trained machine learning model.
It connects to the database, fetches waveform data, and uses a pre-trained model to predict whether the waveform represents a spark event or not.
The program processes the waveform data, extracts features, and evaluates each waveform using the model.
----------------------------------------------------------------------
Structure:

| ml_base_data.py - Machine learning base data file (Fetches data from the database and allows to categorize it, providing feedback for machine learning model input).
| ml_model.py - Machine learning model file (Based on the feedback trains a machine learning model to predict spark events).
|
| main.py - Main program file
|__ Acquisition.py - Acquires data from an oscilloscope and saves it in a MySQL database.
|__ Evaluation.py - Evaluate database data file (Fetches data from the database and evaluates it using the trained machine learning model).
|__ Analysis.py - Analyzes the spark data and generates a report with Graphical and Analytical results.
|_____ MetricAnalysis.py - Provides an analysis on the impact of mechanical defaults on the spark events, based on test metrics (validation).

Environment:
  1. Python 3.12.7
  2. mysql-connector-python 2.2.9
  3. numpy 1.21.0
  4. re 4.3.0
  5. ast 2.0.0
  6. time 4.0.0
  7. joblib 1.2.0

Version: 1.0
Modified on Mar 31th 2025
Author: Maria Rodrigues
"""

import mysql.connector
import numpy as np
import ast
import re
import joblib
import time

# Function to fetch waveform data from the new database and process it
def fetch_data_from_db():
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="sparktest"
    )
    cursor = db.cursor()

    cursor.execute("SELECT id, waveform_data_ch1 FROM oscilloscope_data WHERE id > 2386 and id <= 2886") 
    waveform_ch1 = []
    ids = []
    vertical_scales = []
    horizontal_scales = []
    sampling_periods = []

    for row in cursor.fetchall():
        waveform_str_ch1 = row[1]
        id_value = row[0]

        waveform_str_ch1 = waveform_str_ch1.strip()
        waveform_str_ch1 = re.sub(r"([0-9e+\-.]+)(?=\s+([0-9e+\-.]))", r"\1, ", waveform_str_ch1)

        try:
            waveform_list_ch1 = ast.literal_eval(waveform_str_ch1)
        except Exception as e:
            print(f"Error parsing waveform data for ID {id_value}: {e}")
            continue

        # Ensure waveform has 992 points
        if isinstance(waveform_list_ch1, list) and len(waveform_list_ch1) == 992:
            waveform_ch1.append(waveform_list_ch1)
            ids.append(id_value)

            # Set metadata based on ID
            if 2386 <= id_value <= 2490:
                vertical_scales.append(0.2)  # 200mV for IDs 2386 to 2490
                horizontal_scales.append(50e-9)  # 50ns horizontal scale
                sampling_periods.append(1e9)  # 1 GSa/s
            else:
                vertical_scales.append(5.0)  # 5V for IDs above 2490
                horizontal_scales.append(50e-9)  # 50ns horizontal scale
                sampling_periods.append(1e9)  # 1 GSa/s

        else:
            print(f"Skipping ID {id_value}: waveform length = {len(waveform_list_ch1)}")

    cursor.close()
    db.close()

    return np.array(waveform_ch1), ids, vertical_scales, horizontal_scales, sampling_periods

# Insert waveform data into the second database if spark is detected
def insert_waveform_data(id_value, waveform_data, vertical_scale, horizontal_scale, sampling_period, prediction):
    if prediction == 1:  # Only insert if the prediction is a spark
        db = mysql.connector.connect(
            host="localhost",  # Change if needed
            user="root",  # Your username
            password="",  # Your password
            database="sparktest"  # The new database name
        )
        cursor = db.cursor()

        for point_number, point in enumerate(waveform_data, 1):  # Start from point_number 1
            cursor.execute(""" 
                INSERT INTO sparks (id, point_number, waveform_point, vertical_scale, horizontal_scale, sampling_period)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (id_value, point_number, point, vertical_scale, horizontal_scale, sampling_period))

        db.commit()  # Commit after processing each waveform
        cursor.close()
        db.close()

# Extract features from the waveform data (simple example)
def extract_features_from_waveform(waveform_data):
    if len(waveform_data) == 0:
        return [0, 0, 0, 0]  # Default features for empty waveform

    features = []
    features.append(np.mean(waveform_data))  # Mean of waveform
    features.append(np.std(waveform_data))  # Standard deviation of waveform
    features.append(np.max(waveform_data))  # Maximum value in the waveform
    features.append(np.min(waveform_data))  # Minimum value in the waveform
    return features

# Load the trained model
def load_trained_model():
    model = joblib.load("spark_detection_model.pkl")  # Load the model
    return model

# Evaluate each waveform from the new database using the model
def evaluate_waveforms(waveform_data_list, model):
    predictions = []
    for waveform in waveform_data_list:
        features = extract_features_from_waveform(waveform)
        prediction = model.predict([features])[0]  # Predict if it's a spark (1) or not (0)
        predictions.append(prediction)
    return predictions

# Main function to evaluate the new database and insert sparks
def main():
    # Fetch the waveform data from the database
    waveform_data_list, id_list, vertical_scales, horizontal_scales, sampling_periods = fetch_data_from_db()

    # Load the trained model
    model = load_trained_model()

    # Evaluate waveforms using the model
    predictions = evaluate_waveforms(waveform_data_list, model)

    # Insert the data with spark predictions into the second database
    for i, prediction in enumerate(predictions):
        insert_waveform_data(id_list[i], waveform_data_list[i], vertical_scales[i], horizontal_scales[i], sampling_periods[i], prediction)  # Insert only if spark is detected