import cv2
import time
import datetime
import sys
import torch
import argparse
import os
import warnings
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

# Suppress noisy FutureWarnings from YOLOv5 hub code
warnings.filterwarnings("ignore", category=FutureWarning)

class PersonDetector:
    """
    A class to detect persons in a video stream using a pre-trained
    MobileNet SSD model.
    """
    def __init__(self, confidence_threshold=0.5, influx_config=None, device='cpu'):
        self.confidence_threshold = confidence_threshold
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5m', device=device, trust_repo=True)
        self.model.conf = confidence_threshold
        self.model.classes = [0]  # Filter for 'person' class (class 0)
        self.cap = None
        self.video_writer = None
        
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

    def _initialize_camera(self, source='0'):
        """Initializes the video capture device."""
        print(f"[INFO] Starting video stream from source: {source}...")
        # Try to convert source to int for webcam index, otherwise treat as path/URL
        try:
            source = int(source)
        except ValueError:
            pass
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            print(f"[ERROR] Could not open video source: {source}")
            return False
        # Allow the camera sensor to warm up
        time.sleep(2.0)
        return True

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
                .time(datetime.datetime.now(datetime.timezone.utc))
            try:
                self.write_api.write(bucket=self.influx_config['bucket'], org=self.influx_config['org'], record=point)
            except Exception as e:
                print(f"[ERROR] Failed to write to InfluxDB: {e}")

    def run_detection_loop(self, detection_interval=60, source='0', interval_mode=False, view_img=True, test_mode=False, output_file=None):
        """
        Starts the main loop to capture frames and detect persons periodically.
        """
        if not self._initialize_camera(source):
            print("[ERROR] Initial camera connection failed.")
            return

        if test_mode:
            if not output_file:
                base_name = os.path.basename(str(source))
                output_file = f"output_{base_name}"
                if not output_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    output_file += ".mp4"
            
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or fps is None:
                fps = 30.0
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
            print(f"[INFO] Test mode enabled. Saving output to: {output_file}")

        print("[INFO] Starting detection loop...")
        last_detection_time = 0
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    # avoid a processing loop for video files
                    if os.path.isfile(str(source)):
                        print("[INFO] Video file processing complete.")
                        break

                    print("[WARNING] Stream disconnected. Re-initializing in 5s...")
                    self.cap.release()
                    time.sleep(5)
                    self._initialize_camera(source)
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
                if view_img:
                    cv2.imshow("Person Detection", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if person_detected:
                    current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"Person detected at: {current_time_str} | Count: {person_count}", flush=True)
                    self.save_detection_event(person_count)

                if self.video_writer:
                    self.video_writer.write(display_frame)

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
        if self.video_writer:
            self.video_writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Person Detection Script")
    parser.add_argument("--source", default='0', help="Video source. Can be a webcam index (e.g., '0'), a video file path, or an RTSP stream URL.")
    parser.add_argument("--detection-interval", type=int, default=10, help="Interval in seconds between detections. Set to 0 to detect on every frame.")
    parser.add_argument("--influx-url", default="http://localhost:8086", help="InfluxDB URL")
    parser.add_argument("--influx-token", default="28bcEXQMj8jHC-Jrgqv-YxgUmDIxXPaolDQpXPxazJSl4y2M_UwaxA_p2N1X_xtWi_tD2hAbUjSE6huzKa4KuA==", help="InfluxDB Token")
    parser.add_argument("--influx-org", default="digi", help="InfluxDB Organization")
    parser.add_argument("--influx-bucket", default="persons-detection", help="InfluxDB Bucket")
    parser.add_argument("--no-view", action="store_true", help="Disable the visual display window (useful for headless/docker)")
    parser.add_argument("--device", default="cpu", help="Device to run inference on (e.g., 'cpu' or '0' for cuda)")
    parser.add_argument("--test-mode", action="store_true", help="Enable test mode to save output video with detections.")
    parser.add_argument("--output-file", default=None, help="Output video file path for test mode.")
    parser.add_argument("--save-db", action="store_true", help="Force saving to InfluxDB even when in test mode.")
    args = parser.parse_args()

    # --- Configuration ---
    CONFIDENCE_THRESHOLD = 0.6
    DETECTION_INTERVAL = args.detection_interval  # seconds
    INTERVAL_MODE = DETECTION_INTERVAL > 0
    
    # Skip InfluxDB initialization in test mode unless specifically requested
    influx_config = None
    if not args.test_mode or args.save_db:
        influx_config = {
            'url': args.influx_url,
            'token': args.influx_token,
            'org': args.influx_org,
            'bucket': args.influx_bucket
        }

    # --- Execution ---
    detector = PersonDetector(
        confidence_threshold=CONFIDENCE_THRESHOLD,
        influx_config=influx_config,
        device=args.device
    )
    detector.run_detection_loop(
        detection_interval=DETECTION_INTERVAL,
        interval_mode=INTERVAL_MODE,
        source=args.source,
        view_img=not args.no_view,
        test_mode=args.test_mode,
        output_file=args.output_file
    )
