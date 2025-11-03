"""Video display and annotation utilities."""
import cv2
import numpy as np


class VideoDisplay:
    """Handles video display and annotations."""
    
    @staticmethod
    def draw_text(frame, text, position, font_scale=1, 
                  color=(255, 255, 255), thickness=2):
        """
        Draw text on the frame.
        
        Args:
            frame: Image frame
            text: Text to display
            position: (x, y) tuple for text position
            font_scale: Font size scale
            color: BGR color tuple
            thickness: Text thickness
        """
        cv2.putText(
            frame, 
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX, 
            font_scale, 
            color, 
            thickness, 
            cv2.LINE_AA
        )
    
    @staticmethod
    def draw_angle(frame, angle, position, frame_width, frame_height):
        """
        Draw angle value at a specific landmark position.
        
        Args:
            frame: Image frame
            angle: Angle value to display
            position: Normalized [x, y] position
            frame_width: Width of the frame
            frame_height: Height of the frame
        """
        pixel_pos = tuple(
            np.multiply(position, [frame_width, frame_height]).astype(int)
        )
        VideoDisplay.draw_text(
            frame, 
            str(int(angle)), 
            pixel_pos,
            font_scale=0.7,
            color=(255, 255, 255),
            thickness=2
        )
    
    @staticmethod
    def draw_stats(frame, rep_count, state=None, angle=None):
        """
        Draw statistics overlay on the frame.
        
        Args:
            frame: Image frame
            rep_count: Number of reps
            state: Current state (optional)
            angle: Current angle (optional)
        """
        # Draw rep count
        VideoDisplay.draw_text(
            frame,
            f'Reps: {rep_count}',
            (10, 40),
            font_scale=1,
            color=(0, 255, 0),
            thickness=2
        )
        
        # Draw state if provided
        if state:
            VideoDisplay.draw_text(
                frame,
                f'State: {state}',
                (10, 80),
                font_scale=0.8,
                color=(0, 255, 255),
                thickness=2
            )
        
        # Draw angle if provided
        if angle:
            VideoDisplay.draw_text(
                frame,
                f'Angle: {int(angle)}°',
                (10, 120),
                font_scale=0.8,
                color=(255, 255, 0),
                thickness=2
            )
    
    @staticmethod
    def show_frame(window_name, frame):
        """
        Display a frame in a window.
        
        Args:
            window_name: Name of the window
            frame: Image frame to display
        """
        cv2.imshow(window_name, frame)
        
    @staticmethod
    def save_frame(frame, filename):
        """
        Save a frame to a file.
        
        Args:
            frame: Image frame to save
            filename: Path to save the image
        """
        cv2.imwrite(filename, frame)
    
    @staticmethod
    def create_video_writer(output_path, fps, frame_width, frame_height):
        """
        Create a VideoWriter object for saving video.
        
        Args:
            output_path: Path to save the output video
            fps: Frames per second
            frame_width: Width of the video frames
            frame_height: Height of the video frames
            
        Returns:
            cv2.VideoWriter object
        """
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    @staticmethod
    def write_frame(video_writer, frame):
        """
        Write a frame to the video file.
        
        Args:
            video_writer: cv2.VideoWriter object
            frame: Image frame to write
        """
        video_writer.write(frame)
    
    @staticmethod
    def release_video_writer(video_writer):
        """
        Release the VideoWriter object.
        
        Args:
            video_writer: cv2.VideoWriter object to release
        """
        video_writer.release()
