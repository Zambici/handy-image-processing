import cv2
import time
import datetime
import sys
import torch
import argparse

class PersonDetector:
    """
    A class to detect persons in a video stream using a pre-trained
    MobileNet SSD model.
    """
    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device='cuda:0')
        # self.model = torch.hub.load('ultralytics/yolov5', 'yolov5x', device='cpu')
        self.model.conf = confidence_threshold
        self.model.classes = [0]  # Filter for 'person' class (class 0)
        self.cap = None

    def _initialize_camera(self, camera_index=0):
        """Initializes the video capture device."""
        print("[INFO] Starting video stream...")
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print("[ERROR] Could not open webcam.")
            sys.exit(1)
        # Allow the camera sensor to warm up
        time.sleep(2.0)

    def process_frame(self, frame):
        """
        Processes a single frame to detect persons.
        Returns the annotated frame and True if a person is detected, False otherwise.
        """
        # Convert BGR to RGB for YOLOv5
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(frame_rgb)

        # Check for detections
        # results.xyxy[0] is a tensor of detections for the first image
        person_detected = len(results.xyxy[0]) > 0

        # Render detections on the frame
        results.render()  # Updates results.imgs with boxes
        annotated_frame = results.ims[0]
        # Convert back to BGR for OpenCV display
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

        return person_detected, annotated_frame

    def run_detection_loop(self, detection_interval=60, camera_index=0, interval_mode=False):
        """
        Starts the main loop to capture frames and detect persons periodically.
        """
        self._initialize_camera(camera_index)
        print("[INFO] Starting detection loop...")
        last_detection_time = 0
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("[WARNING] Could not read frame from webcam. Retrying...")
                    time.sleep(1)
                    continue

                should_detect = True
                if interval_mode:
                    if time.time() - last_detection_time < detection_interval:
                        should_detect = False
                    else:
                        last_detection_time = time.time()

                if should_detect:
                    person_detected, annotated_frame = self.process_frame(frame)
                    display_frame = annotated_frame
                else:
                    person_detected = False
                    display_frame = frame
                
                # Visualize detection
                cv2.imshow("Person Detection", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if person_detected:
                    current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"Person detected at: {current_time_str}")

        except KeyboardInterrupt:
            print("\n[INFO] Stopping script.")
        finally:
            self._cleanup()
            cv2.destroyAllWindows()

    def _cleanup(self):
        """Releases the video capture resources."""
        print("[INFO] Cleaning up...")
        if self.cap:
            self.cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Person Detection Script")
    parser.add_argument("--interval-mode", action="store_true", help="Run detection at intervals specified by DETECTION_INTERVAL.")
    args = parser.parse_args()

    # --- Configuration ---
    CONFIDENCE_THRESHOLD = 0.5
    DETECTION_INTERVAL = 20  # 1 minute

    # --- Execution ---
    detector = PersonDetector(
        confidence_threshold=CONFIDENCE_THRESHOLD
    )
    detector.run_detection_loop(detection_interval=DETECTION_INTERVAL, interval_mode=args.interval_mode)
