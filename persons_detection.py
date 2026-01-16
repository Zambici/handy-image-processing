import cv2
import time
import datetime
import sys
import torch
import argparse
import os
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

class PersonDetector:
    """
    A class to detect persons in a video stream using a pre-trained
    MobileNet SSD model.
    """
    def __init__(self, confidence_threshold=0.5, influx_config=None):
        self.confidence_threshold = confidence_threshold
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device='cuda:0')
        # self.model = torch.hub.load('ultralytics/yolov5', 'yolov5x', device='cpu')
        self.model.conf = confidence_threshold
        self.model.classes = [0]  # Filter for 'person' class (class 0)
        self.cap = None
        
        # InfluxDB Initialization
        self.influx_client = None
        self.write_api = None
        self.influx_config = influx_config
        if self.influx_config:
            try:
                self.influx_client = InfluxDBClient(url=self.influx_config['url'], token=self.influx_config['token'], org=self.influx_config['org'])
                self.write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
                print(f"[INFO] Connected to InfluxDB at {self.influx_config['url']}")
            except Exception as e:
                print(f"[ERROR] Failed to connect to InfluxDB: {e}")

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
        Returns the annotated frame and the count of persons detected.
        """
        # Convert BGR to RGB for YOLOv5
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(frame_rgb)

        # Check for detections
        # results.xyxy[0] is a tensor of detections for the first image
        person_count = len(results.xyxy[0])

        # Render detections on the frame
        results.render()  # Updates results.imgs with boxes
        annotated_frame = results.ims[0]
        # Convert back to BGR for OpenCV display
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

        return person_count, annotated_frame

    def save_detection_event(self, count):
        """Saves a detection event to InfluxDB."""
        if self.write_api and self.influx_config:
            point = Point("person_detection") \
                .tag("location", "camera_main") \
                .field("detected", 1) \
                .field("persons_count", count) \
                .time(datetime.datetime.utcnow())
            try:
                self.write_api.write(bucket=self.influx_config['bucket'], org=self.influx_config['org'], record=point)
            except Exception as e:
                print(f"[ERROR] Failed to write to InfluxDB: {e}")

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
                    person_count, annotated_frame = self.process_frame(frame)
                    person_detected = person_count > 0
                    display_frame = annotated_frame
                else:
                    person_detected = False
                    person_count = 0
                    display_frame = frame
                
                # Visualize detection
                cv2.imshow("Person Detection", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if person_detected:
                    current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"Person detected at: {current_time_str} | Count: {person_count}")
                    self.save_detection_event(person_count)

        except KeyboardInterrupt:
            print("\n[INFO] Stopping script.")
        finally:
            self._cleanup()
            cv2.destroyAllWindows()

    def _cleanup(self):
        """Releases the video capture resources."""
        print("[INFO] Cleaning up...")
        if self.influx_client:
            self.influx_client.close()
        if self.cap:
            self.cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Person Detection Script")
    parser.add_argument("--interval-mode", default="10", help="Run detection at intervals specified by DETECTION_INTERVAL.")
    parser.add_argument("--influx-url", default="http://localhost:8086", help="InfluxDB URL")
    parser.add_argument("--influx-token", default="28bcEXQMj8jHC-Jrgqv-YxgUmDIxXPaolDQpXPxazJSl4y2M_UwaxA_p2N1X_xtWi_tD2hAbUjSE6huzKa4KuA==", help="InfluxDB Token")
    parser.add_argument("--influx-org", default="digi", help="InfluxDB Organization")
    parser.add_argument("--influx-bucket", default="persons-detection", help="InfluxDB Bucket")
    args = parser.parse_args()

    # --- Configuration ---
    CONFIDENCE_THRESHOLD = 0.4
    DETECTION_INTERVAL = int(args.interval_mode)  # seconds
    
    influx_config = {
        'url': args.influx_url,
        'token': args.influx_token,
        'org': args.influx_org,
        'bucket': args.influx_bucket
    }

    # --- Execution ---
    detector = PersonDetector(
        confidence_threshold=CONFIDENCE_THRESHOLD,
        influx_config=influx_config
    )
    detector.run_detection_loop(detection_interval=DETECTION_INTERVAL, interval_mode=args.interval_mode)
