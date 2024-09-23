import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def isolation_forest_range(data, feature_name, contamination=0.1):
    """
    Function to calculate the range of values for a feature based on the Isolation Forest algorithm.
    
    Parameters:
    data (pd.DataFrame): The dataset.
    feature_name (str): The column name of the feature to analyze.
    contamination (float): The proportion of outliers in the data set (default is 0.1).
    
    Returns:
    tuple: The lower and upper bounds of the feature based on the Isolation Forest algorithm.
    """
    # Remove rows with null values in the specified feature
    data_cleaned = data.dropna(subset=[feature_name])
    
    # Reshape the data for Isolation Forest (it expects a 2D array)
    X = data_cleaned[feature_name].values.reshape(-1, 1)
    
    # Create and fit the Isolation Forest model
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    iso_forest.fit(X)
    
    # Predict anomalies
    anomaly_labels = iso_forest.predict(X)
    
    # Separate normal and anomaly data points
    normal_data = X[anomaly_labels == 1].flatten()
    
    # Calculate the range based on the normal data points
    lower_bound = np.min(normal_data)
    upper_bound = np.max(normal_data)
    
    return lower_bound, upper_bound


def monitor_server(data, feature_name, normal_range):
    """
    Function to monitor server performance and detect anomalies based on specified normal range.
    
    Parameters:
    data (pd.DataFrame): The dataset with server performance metrics.
    feature_name (str): The column name of the feature to analyze (e.g., CPU Utilization).
    normal_range (tuple): The normal operating range of the feature (lower_bound, upper_bound).
    """
    # Initialize lists to store detected intervals
    unresponsive_intervals = []
    light_load_intervals = []
    high_load_intervals = []

    # Extract lower and upper bounds from the normal_range tuple
    lower_bound, upper_bound = normal_range

    # Initialize flags and start times for different conditions
    in_unresponsive = False
    in_light_load = False
    in_high_load = False
    start_unresponsive = None
    start_light_load = None
    start_high_load = None

    # Iterate through the dataset to detect periods of different conditions
    for i in range(len(data)):
        value = data[feature_name].iloc[i]
        timestamp = data.index[i]

        # Detect unresponsive periods (null values)
        if pd.isnull(value):
            if not in_unresponsive:
                start_unresponsive = timestamp
                in_unresponsive = True
        else:
            if in_unresponsive:
                unresponsive_intervals.append((start_unresponsive, timestamp))
                in_unresponsive = False
            
            # Detect light load periods (values below the lower bound)
            if value < lower_bound:
                if not in_light_load:
                    start_light_load = timestamp
                    in_light_load = True
            else:
                if in_light_load:
                    light_load_intervals.append((start_light_load, timestamp))
                    in_light_load = False
            
            # Detect high load periods (values above the upper bound)
            if value > upper_bound:
                if not in_high_load:
                    start_high_load = timestamp
                    in_high_load = True
            else:
                if in_high_load:
                    high_load_intervals.append((start_high_load, timestamp))
                    in_high_load = False
    
    # Handle any ongoing periods at the end of the dataset
    if in_unresponsive:
        unresponsive_intervals.append((start_unresponsive, data.index[-1]))
    if in_light_load:
        light_load_intervals.append((start_light_load, data.index[-1]))
    if in_high_load:
        high_load_intervals.append((start_high_load, data.index[-1]))
    
    # Generate and print a summary of detected intervals
    print("Summary of Server Behavior:")
    for start, end in unresponsive_intervals:
        print(f"Server was unresponsive from {start} to {end}.")
    for start, end in light_load_intervals:
        print(f"Server was under very light load from {start} to {end}.")
    for start, end in high_load_intervals:
        print(f"Server was under very high load from {start} to {end}.")

    # Plotting the server performance data and detected anomalies
    plt.figure(figsize=(14, 8))
    
    # Plot CPU Utilization for data points within the normal range, breaking the line for unresponsive periods
    normal_data = data[(data[feature_name] >= lower_bound) & (data[feature_name] <= upper_bound)]
    
    # Sort unresponsive intervals and add start and end of data as plot segments
    plot_segments = [(data.index[0], data.index[0])] + unresponsive_intervals + [(data.index[-1], data.index[-1])]
    plot_segments.sort(key=lambda x: x[0])

    for i in range(len(plot_segments) - 1):
        segment_start = plot_segments[i][1]
        segment_end = plot_segments[i+1][0]
        segment_data = normal_data[(normal_data.index >= segment_start) & (normal_data.index <= segment_end)]
        plt.plot(segment_data.index, segment_data[feature_name], color='blue', marker='o', linestyle='-', label='CPU Utilization' if i == 0 else "")
    
    # Plot normal operating range as horizontal lines
    plt.axhline(y=lower_bound, color='green', linestyle='--', label='Lower Bound of Normal Range')
    plt.axhline(y=upper_bound, color='green', linestyle='--', label='Upper Bound of Normal Range')

    # Plot unresponsive periods as shaded areas
    for start, end in unresponsive_intervals:
        plt.axvspan(start, end, color='red', alpha=0.3, label='Unresponsive Period')

    # Plot light load anomalies as orange scatter points
    light_load_points = data[(data[feature_name] < lower_bound) & (data[feature_name].notnull())]
    plt.scatter(light_load_points.index, light_load_points[feature_name], color='orange', label='Light Load Anomaly', marker='x')

    # Plot high load anomalies as purple scatter points
    high_load_points = data[(data[feature_name] > upper_bound) & (data[feature_name].notnull())]
    plt.scatter(high_load_points.index, high_load_points[feature_name], color='purple', label='High Load Anomaly', marker='^')

    plt.title('Server CPU Utilization with Anomalies')
    plt.xlabel('Timestamp')
    plt.ylabel('CPU Utilization (%)')
    plt.legend()
    plt.grid(True)
    plt.show()
