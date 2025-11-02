"""Pose detection using MediaPipe."""
import cv2
import mediapipe as mp
import numpy as np
from utils.angle_calculator import calculate_angle


class PoseDetector:
    """Handles pose detection and landmark extraction."""
    
    def __init__(self, static_image_mode=False, 
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        """
        Initialize the pose detector.
        
        Args:
            static_image_mode: Whether to treat input as static images
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
    
    def process_frame(self, frame):
        """
        Process a frame to detect pose landmarks.
        
        Args:
            frame: BGR image frame
        
        Returns:
            results: MediaPipe pose detection results
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        return results
    
    def get_landmarks(self, results):
        """
        Extract landmarks from results.
        
        Args:
            results: MediaPipe pose detection results
        
        Returns:
            landmarks or None
        """
        if results.pose_landmarks:
            return results.pose_landmarks.landmark
        return None
    
    def get_joint_points(self, landmarks, joint_name):
        """
        Get x, y coordinates for a specific joint.
        
        Args:
            landmarks: Pose landmarks
            joint_name: Name of the joint (e.g., 'RIGHT_SHOULDER')
        
        Returns:
            list: [x, y] coordinates
        """
        joint = getattr(self.mp_pose.PoseLandmark, joint_name)
        return [landmarks[joint.value].x, landmarks[joint.value].y]
    
    def get_arm_angle(self, landmarks, side='RIGHT'):
        """
        Calculate the angle of the arm (shoulder-elbow-wrist).
        
        Args:
            landmarks: Pose landmarks
            side: 'RIGHT' or 'LEFT'
        
        Returns:
            float: Angle in degrees
        """
        shoulder = self.get_joint_points(landmarks, f'{side}_SHOULDER')
        elbow = self.get_joint_points(landmarks, f'{side}_ELBOW')
        wrist = self.get_joint_points(landmarks, f'{side}_WRIST')
        
        return calculate_angle(shoulder, elbow, wrist)
    
    def draw_landmarks(self, frame, results):
        """
        Draw pose landmarks on the frame.
        
        Args:
            frame: Image frame to draw on
            results: MediaPipe pose detection results
        """
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS
            )
    
    def close(self):
        """Release resources."""
        self.pose.close()
