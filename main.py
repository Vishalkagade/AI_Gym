"""Main application for bicep curl counter using pose estimation."""
import cv2
from core import PoseDetector, RepCounter
from ui import VideoDisplay
from utils import CSVDataLogger


def main():
    """Main function to run the bicep curl counter."""
    # Initialize components
    pose_detector = PoseDetector(static_image_mode=False)
    rep_counter = RepCounter(up_threshold=160, down_threshold=70)
    video_display = VideoDisplay()
    csv_logger = CSVDataLogger(filename='bicep_curl_data.csv', append=False)
    
    print("Initialization complete. Starting video capture...")
    print("Press ESC to exit")
    
    # Start video capture
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Process the frame for pose detection
            results = pose_detector.process_frame(frame)
            
            # Get landmarks and calculate angle
            landmarks = pose_detector.get_landmarks(results)
            
            angle = None  # Initialize angle variable
            
            if landmarks:
                # Get right arm angle
                angle = pose_detector.get_arm_angle(landmarks, side='RIGHT')
                
                # Update rep counter
                rep_counter.update(angle)
                
                # Get elbow position for angle annotation
                elbow_pos = pose_detector.get_joint_points(landmarks, 'RIGHT_ELBOW')
                
                # Draw angle at elbow position
                video_display.draw_angle(
                    frame, 
                    angle, 
                    elbow_pos,
                    frame.shape[1], 
                    frame.shape[0]
                )
                
                # Draw pose landmarks
                pose_detector.draw_landmarks(frame, results)

            # Log data to CSV for every frame
            csv_logger.log_frame(
                rep_count=rep_counter.get_count(),
                state=rep_counter.get_state(),
                angle=angle,
                pose_detected=landmarks is not None
            )

            # Draw statistics overlay
            video_display.draw_stats(
                frame,
                rep_counter.get_count(),
                state=rep_counter.get_state(),
                angle=angle if landmarks else None
            )
            
            # Display the frame
            video_display.show_frame('Bicep Curl Counter', frame)
            
            # Exit on ESC key
            if cv2.waitKey(5) & 0xFF == 27:
                break
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        pose_detector.close()
        csv_logger.close()
        print(f"\nSession complete! Total reps: {rep_counter.get_count()}")


if __name__ == "__main__":
    main()
