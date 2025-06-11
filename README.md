# Dry Tests on Bosch Gas Boilers
Repository within the scope of the master's dissertation in Mechanical Engineering entitled "Dry Tests on Bosch Gas Boilers", by Maria Rodrigues.

The dry test is divided into two specific tests: the spark test and the leak test.

As such, the files necessary for both tests are provided in this repository.



**Spark Test Structure:**

| ml_base_data.py - Machine learning base data file (Fetches data from the database and allows to categorize it, providing feedback for machine learning model input).

| ml_model.py - Machine learning model file (Based on the feedback trains a machine learning model to predict spark events).

|

| main.py - Main program file

|__ Acquisition.py - Acquires data from an oscilloscope and saves it in a MySQL database.

|__ Evaluation.py - Evaluate database data file (Fetches data from the database and evaluates it using the trained machine learning model).

|__ Analysis.py - Analyzes the spark data and generates a report with Graphical and Analytical results.

|_____ MetricAnalysis.py - Provides an analysis on the impact of mechanical defaults on the spark events, based on test metrics (validation).



**Leak Test Structure:**

| AnalysisTool.py - Runs the leak analysis on previously stored files, through the fetching system via user input.

|

| LeakCode.py - Runs real-time leak analysis after thermal video capturing.
