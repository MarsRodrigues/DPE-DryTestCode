""""
Program name: Acquisition.py
----------------------------------------------------------------------
Program elaborated to acquire data from an oscilloscope and save it in a MySQL database.
----------------------------------------------------------------------
Description:
This program connects to an oscilloscope via a serial port, sends SCPI commands to acquire waveform data, and saves the data in a csv.
It checks if the channel is active before acquiring data.
The program runs in a loop until the user presses 'x' to stop it.
Reads the csv and adds the data to a MySQL database.
It also generates real-time plots of the acquired data and saves them as PNG files. 
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
  3. matplotlib 3.4.3
  4. numpy 1.21.0
  5. pyserial 3.5
  6. keyboard 0.13.5
  7. datetime 4.3.0
  8. csv 1.0.0
  9. os 0.1.0    
  
Version: 1.0
Modified on Mar 31th 2025
Author: Maria Rodrigues
"""

import mysql.connector
import serial
import time
import csv
import keyboard
import matplotlib 
import os
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

#1000 points 200mv 50ns para eletródo dentro
#1000 points 5v 50ns para eletródo dentro

# MySQL connection details
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="sparktest"
)
cursor = db.cursor()

# Create table if it doesn't exist (run this once)
cursor.execute("""
CREATE TABLE IF NOT EXISTS oscilloscope_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME,
    command VARCHAR(255),
    vert_pos_ch1 DOUBLE,
    vert_scale_ch1 DOUBLE,
    waveform_data_ch1 TEXT,
    metadata_ch1 TEXT
)
""")
db.commit()

# Initialize constants and configure COM port
ser = serial.Serial('COM7', 9600, timeout=1)
csv_filename = 'oscilloscope_data.csv'

# Create a directory for saved graphs if it doesn't exist
graphs_dir = 'saved_graphs'
if not os.path.exists(graphs_dir):
    os.makedirs(graphs_dir)

# Initialize the graph counter
graph_id = 1

# Function to get the next auto-incremented ID for the graph filenames
def get_next_graph_id():
    global graph_id
    while os.path.exists(os.path.join(graphs_dir, f'graph_{graph_id}.png')):
        graph_id += 1
    return graph_id

# Write CSV headers if file is empty
def initialize_csv():
    try:
        with open(csv_filename, 'a', newline='') as f:
            if f.tell() == 0:
                csv.writer(f).writerow(['timestamp', 'command', 'vert_pos_ch1', 'vert_scale_ch1', 'waveform_data_ch1', 'metadata_ch1'])
    except Exception as e:
        print(f"CSV Init Error: {e}")

# Append data to CSV
def append_to_csv(data):
    try:
        with open(csv_filename, 'a', newline='') as f:
            csv.writer(f).writerow(data)
    except Exception as e:
        print(f"CSV Write Error: {e}")

# Send SCPI command and get response
def send_command(command):
    try:
        ser.write(f'{command}\n'.encode())
        time.sleep(0.05)
        return ser.read(ser.inWaiting())
    except Exception as e:
        print(f"Command Error: {e}")
        return None

# Decode waveform data from raw bytes
def decode_waveform_data(raw_data):
    try:
        if b"Waveform" in raw_data:
            metadata, waveform_data = raw_data.split(b"Waveform", 1)
            metadata = metadata.decode('utf-8', 'replace')
        else:
            metadata, waveform_data = "Unknown", raw_data

        if len(waveform_data) % 2:  # Trim odd byte length
            waveform_data = waveform_data[:-1]

        decoded_values = np.frombuffer(waveform_data, dtype='<i2')  # Use numpy to decode int16
        scaling_factor = 8 / 65535  # Simplified scaling factor
        decoded_values = (decoded_values + 32768) * scaling_factor - 4

        return metadata, decoded_values[11:1011] 
    except Exception as e:
        print(f"Decoding Error: {e}")
        return "Error", []

# Check if the channel is active
def is_channel_active(channel_num):
    try:
        response = send_command(f":CHAN{channel_num}:DISP?")
        
        # If response is empty or None, treat it as an error
        if not response:
            return False
        
        # Try decoding the response and check if the channel is on
        decoded_response = response.decode('utf-8', errors='ignore').strip()  # Ignore decoding errors
        return decoded_response == 'ON'
    except Exception as e:
        print(f"Channel {channel_num} Check Error: {e}")
        return False

# Initialize CSV and plot
initialize_csv()
fig, ax = plt.subplots(figsize=(10, 8))
line_ch1, = ax.plot([], [], label="Channel 1", color='blue')
ax.set(xlabel='Time (s)', ylabel='Amplitude (V)', title='Real-time Oscilloscope Data - Channel 1')
ax.legend(loc='upper left')

# Buffer to store all recorded data for plotting after the loop
data_buffer = []
periods = []

def main():
    # Main loop for real-time data acquisition and plotting
    try:
        while not keyboard.is_pressed('x'):
            cursor.execute("SELECT MAX(id) FROM oscilloscope_data")
            result = cursor.fetchone()
            start_id = result[0] if result[0] else 0

            if is_channel_active(1):
                raw_data_ch1 = send_command(':ACQ1:MEM?')
                if raw_data_ch1:
                    metadata_ch1, decoded_data_ch1 = decode_waveform_data(raw_data_ch1)
                    append_to_csv([time.strftime("%Y-%m-%d %H:%M:%S"), ':ACQ1:MEM?', 0, 1, str(decoded_data_ch1), metadata_ch1])
                    
                    # Add the decoded data to the buffer
                    data_buffer.append(decoded_data_ch1)
                    
                    # Update plot
                    line_ch1.set_data(np.arange(len(decoded_data_ch1)), decoded_data_ch1)
                    ax.relim()
                    ax.autoscale_view()
                    plt.draw()
                    plt.pause(0.01)
    finally:
        ser.close()
        
        # Record the start time of the current session
        start_time = datetime.now()

        # Read the CSV and insert only new rows added during this session into the MySQL database
        try:
            with open(csv_filename, mode='r') as file:
                reader = csv.reader(file)
                next(reader)  # Skip the header row
                
                for row in reader:
                    row_timestamp = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
                    if row_timestamp >= start_time:
                        cursor.execute("""
                            INSERT INTO oscilloscope_data (timestamp, command, vert_pos_ch1, vert_scale_ch1, waveform_data_ch1, metadata_ch1)
                            VALUES (%s, %s, %s, %s, %s, %s)
                        """, row)
                        db.commit()

        except Exception as e:
            print(f"Error reading CSV or writing to database: {e}")

        # Plot and save all recorded waveforms from the buffer to PNG files
        try:
            for idx, waveform_data in enumerate(data_buffer, start=1):
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.plot(np.arange(len(waveform_data)), waveform_data, label="Channel 1", color='blue')
                ax.set(xlabel='Time (s)', ylabel='Amplitude (V)', title=f'Oscilloscope Data - Channel 1 (Waveform {idx})')
                ax.legend(loc='upper left')
                
                # Get the next available graph ID and save the plot as a PNG
                graph_filename = os.path.join(graphs_dir, f'graph_{get_next_graph_id()}.png')
                plt.savefig(graph_filename)
                plt.close()  # Close the plot to free memory

                print(f"Saved waveform {idx} as {graph_filename}")

        except Exception as e:
            print(f"Error saving waveforms as PNG: {e}")
        
        
        cursor.execute("SELECT MAX(id) FROM oscilloscope_data")
        result = cursor.fetchone()
        end_id = result[0] if result[0] else start_id

        # Append the new period to the global periods list
        periods.append((start_id + 1, end_id))
        print(f"New data period recorded: ({start_id + 1}, {end_id})")
        print(f"All tests ID sets acquired so far: {periods}")

        # Close the database connection
        cursor.close()
        db.close()

        elapsed_time = time.time() - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        
if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
