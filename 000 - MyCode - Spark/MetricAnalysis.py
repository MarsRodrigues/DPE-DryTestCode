""""
Program name: MetricAnalysis.py
----------------------------------------------------------------------
Program elaborated to analyse metrics from the spark events - Validation only.
----------------------------------------------------------------------
Description:
Python program to analyze metrics, in order to evaluate the impact of mechanical defaults on the spark events.
It generates plots for different metrics such as amplitude, period, average, median, and number of peaks, comparing tests.
The tests are categorized into two groups: with filter and without filter, and aim to portray different common faults.
The program uses the seaborn and matplotlib libraries for plotting.
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
  2. matplotlib 3.4.3
  3. os 0.1.0
  4. warnings 4.0.3
  5. seaborn 0.11.2

Version: 1.0
Modified on Mar 31th 2025
Author: Maria Rodrigues
"""

import warnings
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

warnings.filterwarnings("ignore", category=RuntimeWarning)

########################## Defining Plot Settings ########################## 
 
# Set the color palette
blue_palette = sns.color_palette("Blues", 8)
red_palette = sns.color_palette("Reds", 8)
green_palette = sns.color_palette("Greens", 8)
grey_palette = sns.color_palette("Greys", 8)
 
custom_palette = [blue_palette[6], blue_palette[2], red_palette[5], red_palette[2], green_palette[5], green_palette[2], grey_palette[7], grey_palette[2]]
 
# Set the style
sns.set_style("ticks")
 
# Set default figure size
plt.rcParams['figure.figsize'] = (12, 8)
 
# Set text sizes
plt.rcParams['axes.titlesize'] = 36
plt.rcParams['axes.labelsize'] = 34
plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['ytick.labelsize'] = 30
plt.rcParams['legend.fontsize'] = 30
plt.rcParams['font.size'] = 30
plt.rcParams['axes.linewidth'] = 1.0
 
# Use LaTeX for rendering text
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'CMU'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

save_path = r"C:\Users\Maria Rodrigues\Desktop\Investigação ILLIANCE BOSCH\003 - Lab Tests\002 - Spark Test 19.03.25\PDFs"

# Plot all waveforms exactly as they are
def plot(metric_data, test_labels, metric_name, dataset_labels):
    if metric_name == 'Number of Peaks':
        unit = None
    elif metric_name == 'Period':
        unit = 'ns'
    else:
        unit = 'V'
    
    plt.figure(figsize=(12, 8))
    
    for i, data_series in enumerate(metric_data):
        plt.plot(test_labels, data_series, label=dataset_labels[i], linewidth=2, color=custom_palette[i % len(custom_palette)])
    
    plt.xlabel("Test")
    plt.ylabel(f"{metric_name} ({unit})" if unit else metric_name)
    plt.title(f"{metric_name} per Test")
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=24, frameon=False)
    plt.grid(True)
    plt.tight_layout()

    filename_base = f"metric_{metric_name.replace(' ', '_')}"
    plt.savefig(f"{filename_base}.png")
    # plt.savefig(os.path.join(save_path, f"{filename_base}.pdf"), dpi=300)

    plt.close()
    print(f"{metric_name} graph saved.")

def main():
    # Manually set the data for each metric for each test
    # The data is structured as a list of tuples, where each tuple represents a test
    ave=[(0.00192,0.28009,0.30556,0.28932,0.28953,0.29478,0.28569,0.38771), (
        0.00187,0.31096,0.31096,0.28388,0.26707,0.28718,0.26964,0.38666), (
        0.00191,0.25161,0.27895,0.23367,0.25607,0.27228,0.26827,0.00000), (
        0.00182,0.25887,0.25887,0.22526,0.25364,0.27242,0.26404,0.00000 
    )]
    
    md = [ (0.00191, 0.36924, 0.37639, 0.37530, 0.37081, 0.37730, 0.37584, 0.38249),
    (0.00191, 0.37673, 0.37673, 0.37507, 0.36872, 0.37507, 0.37507, 0.38272),
    (0.00191, 0.36633, 0.37566, 0.28132, 0.36900, 0.37556, 0.37537, 0.00000),
    (0.00190, 0.37283, 0.37283, 0.37507, 0.36955, 0.37507, 0.37507, 0.00000)
    ]
    
    npk = [
    (122.42857, 68.36441, 84.57042, 48.28571, 62.20455, 60.67857, 69.94262, 94.82500),
    (121.54737, 76.36170, 76.36170, 45.98113, 58.48438, 61.26866, 65.32258, 106.34694),
    (117.18182, 43.96774, 63.93396, 10.00000, 43.94030, 46.42857, 56.64423, 0.00000),
    (113.78000, 45.91071, 45.91071, 56.50000, 44.39216, 50.27273, 55.25862, 0.00000)
    ]

    amp = [
    (0.00140, 1.90240, 1.48999, 0.63208, 1.18169, 1.31299, 3.23863, 0.03125),
    (0.00143, 1.46918, 1.46918, 0.68930, 1.61342, 1.47915, 3.35904, 0.03125),
    (0.00192, 2.40540, 1.97686, 1.59390, 1.84390, 1.63656, 3.79375, 0.00000),
    (0.00217, 2.44491, 2.44491, 1.54702, 1.87209, 1.66776, 3.57507, 0.00000)
    ]

    
    per = [
    (0.97995, 0.96398, 0.94614, 0.92032, 0.95276, 0.95005, 0.93025, 0.96211),
    (0.97579, 0.96268, 0.96268, 0.89068, 0.95511, 0.95678, 0.93002, 0.94816),
    (0.98005, 0.63471, 0.66199, 0.56400, 0.77009, 0.79008, 0.73970, 0.00000),
    (0.97004, 0.64502, 0.64502, 0.96550, 0.76304, 0.79978, 0.72634, 0.00000)
    ]

    # Test labels for graph ploting
    type = ['Without Filter - Continuous Press', 'Without Filter - Multiple Press', 'With Filter - Continuous Press', 'With Filter - Multiple Press']
    test =['ref', 1, 2, 3, 4, 5, 6, 7]

    plot(amp, test, 'Amplitude', type)
    plot(per, test, 'Period', type)
    plot(ave, test, 'Average', type)
    plot(md, test, 'Median', type)
    plot(npk, test, 'Number of Peaks', type)