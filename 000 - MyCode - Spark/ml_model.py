""""
Program name: ml_model.py
----------------------------------------------------------------------
Program for machine learning model creation and training.
----------------------------------------------------------------------
Description:
Python program used creates a machine learning model to predict spark events based on waveform data.
It fetches waveform data from a MySQL database, reads feedback data from a file, and trains a Random Forest classifier on the data.
The program evaluates the model on a test set and prints the classification report.
The trained model is saved to a file for later use.
The program uses the scikit-learn library for machine learning and joblib for saving the model.
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
  2. Matplotlib 3.7.0
  3. Numpy 1.26.4
  4. MySQL-Connector 9.1.0

AST and RE are a part of the Python library.

Version: 1.0
Modified on Jan 30th 2025
Author: Maria Rodrigues
"""

import mysql.connector
import numpy as np
import random
import ast
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib
import time

def fetch_data_from_db():
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="sparktest"
    )
    cursor = db.cursor()

    cursor.execute("SELECT id, waveform_data_ch1 FROM oscilloscope_data WHERE id >= 2386 AND id <= 2886")
    waveform_ch1 = []
    ids = []

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

        #Ensure waveform has 1000 points
        if isinstance(waveform_list_ch1, list) and len(waveform_list_ch1) == 992:
            waveform_ch1.append(waveform_list_ch1)
            ids.append(id_value)
        else:
            print(f"Skipping ID {id_value}: waveform length = {len(waveform_list_ch1)}")

    cursor.close()
    db.close()

    return np.array(waveform_ch1), ids


# Read feedback from the feedback file
def read_feedback_file(file_path):
    feedback_data = []  # List to store feedback (1 for spark, 0 for no spark)

    # Read the feedback file (assuming it's a CSV or tab-delimited file)
    with open(file_path, 'r') as f:
        for line in f:
            # Parse each line: assume it's in the format: ID, feedback
            parts = line.strip().split(',')
            if len(parts) == 2:
                id_value = int(parts[0])  # ID of the waveform
                feedback = int(parts[1])  # Feedback (1 for spark, 0 for no spark)
                feedback_data.append((id_value, feedback))

    return feedback_data


# Extract features from the waveform data (you can add more features here)
def extract_features_from_waveform(waveform_data):
    features = []
    features.append(np.mean(waveform_data))  # Mean of waveform
    features.append(np.std(waveform_data))  # Standard deviation of waveform
    features.append(np.max(waveform_data))  # Maximum value in the waveform
    features.append(np.min(waveform_data))  # Minimum value in the waveform
    return features


# Prepare data for training
def prepare_data(waveform_ch1, feedback_data, ids):
    X = []  # Features
    y = []  # Labels (spark or not)
    
    for id_value, feedback in feedback_data:
        # Find the waveform corresponding to the id_value
        if id_value in ids:
            waveform = waveform_ch1[ids.index(id_value)]  # Fetch waveform based on the id
            features = extract_features_from_waveform(waveform)
            X.append(features)
            y.append(feedback)
    
    return np.array(X), np.array(y)


def main():
    # Fetch waveform data and IDs from the database
    waveform_ch1, ids = fetch_data_from_db()

    # Read feedback data from the feedback file
    feedback_data = read_feedback_file('feedback_data.txt')  # Format: (id, label)

    # Randomly select 1/3 of the feedback data for training and 2/3 for evaluation
    random.shuffle(feedback_data)
    split_idx = len(feedback_data) // 3
    feedback_train = feedback_data[:split_idx]
    feedback_test = feedback_data[split_idx:]

    # Prepare training data
    X_train, y_train = prepare_data(waveform_ch1, feedback_train, ids)

    # Train the model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Prepare evaluation data
    X_test, y_test = prepare_data(waveform_ch1, feedback_test, ids)

    # Evaluate the model
    y_pred = clf.predict(X_test)
    print("\n Evaluation on test data:")
    print(classification_report(y_test, y_pred))

    # Save the trained model
    joblib.dump(clf, "spark_detection_model.pkl")


if __name__ == "__main__":
    start_time = time.time()
    
    main()
    
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")