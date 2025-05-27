""""
Program name: ml_base_data.py
----------------------------------------------------------------------
Program for fetching data from the database and categorizing it for machine learning model input.
----------------------------------------------------------------------
Description:
Python program used to fetch waveform data from the database, plot each waveform, and asks the user to provide feedback on whether the waveform represents a spark event or not.
The feedback data is then saved to a file for later use in training the machine learning model.
The feedback data consists of waveform data and corresponding labels (1 for spark, 0 for no spark).
The program is part of a larger system that aims to predict spark events based on oscilloscope data.
This categorization visual method makes it easier for the user to provide feedback on the data.
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

import numpy as np
import mysql.connector
import ast
import re
import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def fetch_data_from_db():
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="sparktest"
    )
    cursor = db.cursor()

    cursor.execute("SELECT id, waveform_data_ch1 FROM oscilloscope_data WHERE id > 2623 and id <= 2886") 
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

        if isinstance(waveform_list_ch1, list) and len(waveform_list_ch1) == 992:
            waveform_ch1.append(waveform_list_ch1)
            ids.append(id_value)
        else:
            print(f"Skipping ID {id_value}: waveform length = {len(waveform_list_ch1)}")

    cursor.close()
    db.close()

    return np.array(waveform_ch1), ids

# Function to plot each waveform and ask for feedback
def plot_waveform_and_get_feedback(waveform, id_value):
    plt.figure(figsize=(10, 6))
    x_data = np.arange(len(waveform))  # Generate x-axis data for the 1000 points
    plt.plot(x_data, waveform, label=f"Waveform ID: {id_value}")
    plt.title(f"Waveform with ID {id_value}")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend(loc='upper right')
    plt.show()

    # Ask for feedback: 1 for spark, 0 for no spark
    feedback = input(f"Is this a spark? (1 for yes, 0 for no) for ID {id_value}: ")
    
    if feedback != '0' and feedback != '1':
        print("Invalid feedback. Please enter 0 for no spark or 1 for spark.")
        feedback = input(f"Is this a spark? (1 for yes, 0 for no) for ID {id_value}: ")
        0
    return int(feedback)  # Convert feedback to integer

# Fetch and analyze waveform data
def collect_feedback_for_model(waveform_ch1, ids):
    feedback_data = []  # List to store feedback (1 for spark, 0 for no spark)
    
    for i , waveform in enumerate(waveform_ch1):
        id_value = ids[i]
        feedback = plot_waveform_and_get_feedback(waveform, id_value)
        feedback_data.append((id_value, feedback))  # Store ID and feedback

    return feedback_data

# Fetch data from the database
waveform_ch1, ids = fetch_data_from_db()

# Collect feedback from the user (you will manually classify waveforms as spark or not)
feedback_data = collect_feedback_for_model(waveform_ch1, ids)

# Save feedback data to a file for later use in training
with open("feedback_data.txt", "w") as f:
    for id_value, feedback in feedback_data:
        f.write(f"{id_value},{feedback}\n")

print("Feedback data collection complete.")
