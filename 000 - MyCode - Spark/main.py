""""
Program name: main.py
----------------------------------------------------------------------
Main program file for the spark event detection system.
----------------------------------------------------------------------
Description:
This program orchestrates the entire process of acquiring data from an oscilloscope, evaluating it using a machine learning model, and analyzing the results.
It imports and calls the necessary modules for data acquisition, evaluation, and analysis.
Both ml_base_data.py and ml_model.py are ran beforehand to create and train the model. 
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

Version: 1.0
Modified on Mar 31th 2025
Author: Maria Rodrigues
"""

import Acquisition
import Analysis
import Evaluation
import time

# Timer for acquisition
start_time = time.time()

# Acquisition script call
# Acquires data from the oscilloscope and inserts it into a first table in the database
# The data is acquired in real-time and saved in a CSV file
Acquisition.main()

elapsed_time = time.time() - start_time
print(f"Elapsed time for Acqusition: {elapsed_time:.2f} seconds")

# Restart timer for the data evaluation
start_time = time.time()

# Evaluate script call
# Separates sparks from noise and insert the filtered data into a second table in the database
# The data is evaluated using a trained machine learning model

Evaluation.main()

elapsed_time = time.time() - start_time
print(f"Elapsed time for Evaluation: {elapsed_time:.2f} seconds")

# Restart timer for the analysis
start_time = time.time()

# Analysis script call
# Analyzes the data from the second table in the database and generates a report with Graphical and Analytical results
# The report is saved in CSV files and graphs are generated and saved as PNG files
Analysis.main()

elapsed_time = time.time() - start_time
print(f"Elapsed time for Analysis: {elapsed_time:.2f} seconds")


"""
 Metric Analysis
 
 Metric graphs for the reference validation tests are generated and saved as PNG files
 This analysis was only conducted for the reference tests to evaluate the impact of mechanical defaults on the spark events
 Reference test periods analysed for the below types of tests:
 
 periods = [(2386, 4000), (2386, 2490), (2491, 2585), (2725, 2842), (2843, 2936), (3619, 3760), (3761, 3814), (3815, 3947), (3948, 4000), (3217, 3348), (3349, 3412), (2937, 3020), (3021, 3087), (3413, 3534), (3535, 3596),  (3088, 3167), (3168, 3216)]
 type = ['Without Filter - Continuous Press', 'Without Filter - Multiple Press', 'With Filter - Continuous Press', 'With Filter - Multiple Press']
 test =['ref', 1, 2, 3, 4, 5, 6, 7]

    import MetricAnalysis

        # Restart timer for the metric analysis
        start_time = time.time()

        MetricAnalysis.main()

        elapsed_time = time.time() - start_time
        print(f"Elapsed time for Metrics: {elapsed_time:.2f} seconds")

"""