import argparse
import cv2
import os
from ultralytics import YOLO

def run_inference(model_path, source, conf_threshold=0.25):
    """
    Runs inference using a YOLO model on a specified source.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at '{model_path}'.")
        print("Please train the model first or check the path.")
        return

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # Check if source is an image file based on extension
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    is_image = any(source.lower().endswith(ext) for ext in image_exts)

    if is_image:
        if not os.path.exists(source):
            print(f"Error: Image file '{source}' not found.")
            return
        
        print(f"Processing image: {source}")
        results = model(source, conf=conf_threshold)
        
        # Visualize
        for r in results:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            cv2.imshow("YOLO Detection", im_array)
            print("Press 'q' to close...")
            while True:
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break

        cv2.destroyAllWindows()

    else:
        # Assume video source (webcam index or file/stream path)
        video_source = source
        if source.isdigit():
            video_source = int(source)
            print(f"Using webcam index: {video_source}")
        else:
            print(f"Using video source: {video_source}")

        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error: Could not open video source '{source}'.")
            return

        print("Starting inference. Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of stream.")
                break

            # Run inference on the frame
            results = model(frame, conf=conf_threshold)

            # Process results
            for r in results:
                im_array = r.plot()
                cv2.imshow("YOLO Detection", im_array)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test trained YOLO model.")
    parser.add_argument("--model", type=str, default="runs/detect/dataset_pumpkin_yolox/weights/best.pt", 
                        help="Path to the trained model weights (.pt file)")
    parser.add_argument("--source", type=str, required=True, 
                        help="Input source: '0' for webcam, 'rtsp://...' for stream, or path to image/video file.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for detections.")
    
    args = parser.parse_args()
    run_inference(args.model, args.source, args.conf)
