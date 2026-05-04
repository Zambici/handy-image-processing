import cv2
import numpy as np
import os
import sys

def detect_and_read_qr(original):
    """
    Detects and decodes QR codes using OpenCV's built-in detector.
    """
    detector = cv2.QRCodeDetector()
    
    # detectAndDecodeMulti handles multiple QR codes in a single frame.
    # It returns:
    # - retval: True if at least one QR code is detected
    # - decoded_info: List of strings containing the decoded data
    # - points: List of arrays containing the 4 corner points of each detected QR
    # - straight_qrcode: Rectified and binarized QR code images (not used here)
    retval, decoded_info, points, _ = detector.detectAndDecodeMulti(original)

    if retval:
        for i, data in enumerate(decoded_info):
            if not data:
                # Sometimes a QR is localized but decoding fails due to quality/noise
                continue
                
            # Convert points to integers for drawing (OpenCV returns float32)
            qr_points = points[i].astype(int)
            
            # Draw the bounding polygon around the QR code
            for j in range(len(qr_points)):
                pt1 = tuple(qr_points[j])
                pt2 = tuple(qr_points[(j + 1) % len(qr_points)])
                cv2.line(original, pt1, pt2, (0, 255, 0), 3)
            
            # Label the detection with the decoded string
            # Position the label slightly above the first corner point
            label_pos = (qr_points[0][0], qr_points[0][1] - 10)
            cv2.putText(original, data, label_pos, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            print(f"Detected QR Code: {data}")

    return original

def run_inference(source, frame_step=1):
    """
    Handles logic for images, video files, RTSP streams, or webcam indices.
    """
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    is_image = isinstance(source, str) and os.path.isfile(source) and any(source.lower().endswith(ext) for ext in image_extensions)

    if is_image:
        img = cv2.imread(source)
        if img is None:
            print(f"[ERROR] Could not read image: {source}")
            return
        
        processed_img = detect_and_read_qr(img)
        cv2.imshow("QR Detection", processed_img)
        print("Image processed. Press any key to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        # Handle numeric index for webcams or string path/URL for files/streams
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video source: {source}")
            return

        print(f"[INFO] Starting QR detection on {source}. Press 'q' to quit.")
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            # Only process every Nth frame to optimize performance on streams
            if frame_count % frame_step != 0:
                continue
            
            processed_frame = detect_and_read_qr(frame)
            
            # Resize display for high-resolution sources (like 4K streams)
            display_frame = processed_frame
            if processed_frame.shape[1] > 1280:
                scale = 1280.0 / processed_frame.shape[1]
                display_frame = cv2.resize(processed_frame, (0, 0), fx=scale, fy=scale)
                
            cv2.imshow("QR Detection", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Detect and read QR codes from various sources.")
    parser.add_argument("--source", type=str, required=True, 
                        help="Path to image, video file, RTSP URL (rtsp://...), or camera index (e.g. '0').")
    parser.add_argument("--step", type=int, default=1, 
                        help="Process every Nth frame for video/streams (default: 1).")
    
    args = parser.parse_args()
    run_inference(args.source, frame_step=args.step)