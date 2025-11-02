"""CSV data logging utilities."""
import csv
import os
from datetime import datetime


class CSVDataLogger:
    """Handles logging pose estimation data to CSV file."""
    
    def __init__(self, filename='bicep_curl_data.csv', append=False):
        """
        Initialize the CSV data logger.
        
        Args:
            filename: Name of the CSV file
            append: If True, append to existing file; if False, create new file
        """
        self.filename = filename
        self.file_handle = None
        self.csv_writer = None
        self.frame_count = 0
        
        # Determine mode
        mode = 'a' if append and os.path.exists(filename) else 'w'
        
        # Open file
        self.file_handle = open(filename, mode=mode, newline='')
        self.csv_writer = csv.writer(self.file_handle)
        
        # Write header if new file
        if mode == 'w':
            self.write_header()
    
    def write_header(self):
        """Write CSV header row."""
        header = [
            'frame_number',
            'timestamp',
            'rep_count',
            'state',
            'angle',
            'pose_detected'
        ]
        self.csv_writer.writerow(header)
        self.file_handle.flush()
    
    def log_frame(self, rep_count, state, angle=None, pose_detected=True):
        """
        Log data for a single frame.
        
        Args:
            rep_count: Current number of reps
            state: Current state ('up' or 'down')
            angle: Current arm angle in degrees (None if no pose detected)
            pose_detected: Whether pose was detected in this frame
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        row = [
            self.frame_count,
            timestamp,
            rep_count,
            state,
            f"{angle:.2f}" if angle is not None else "N/A",
            pose_detected
        ]
        
        self.csv_writer.writerow(row)
        self.frame_count += 1
        
        # Flush every 30 frames to ensure data is written
        if self.frame_count % 30 == 0:
            self.file_handle.flush()
    
    def close(self):
        """Close the CSV file."""
        if self.file_handle:
            self.file_handle.flush()
            self.file_handle.close()
            print(f"Data saved to {self.filename} ({self.frame_count} frames)")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
