"""Utility functions for angle calculations."""
import numpy as np


def calculate_angle(a, b, c):
    """
    Calculate the angle between three points.
    
    Args:
        a: First point (shoulder)
        b: Mid point (elbow)
        c: End point (wrist)
    
    Returns:
        float: Angle in degrees
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def save_csv_data(data, filename):
    """
    Save data to a CSV file.
    
    Args:
        data: List of data rows (each row is a list of values)
        filename: Name of the CSV file to save
    """
    import csv

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)