import cv2
import numpy as np
import pytesseract
import os
import sys

# NOTE: If tesseract is not in your PATH, specify it here:
# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

def detect_and_read_numbers(original, show_steps=False, processing_width=1280):
    """
    Detects black numbers on white background surrounded by double borders.
    """
    # 1. Resize for performance if the image is too large (like 4K)
    # Detecting contours on 4K is very slow; we scale down for the detection phase.
    h_orig, w_orig = original.shape[:2]
    scale = 1.0
    # if w_orig > processing_width:
    #     scale = processing_width / float(w_orig)
    #     img_for_det = cv2.resize(original, (0, 0), fx=scale, fy=scale)
    # else:
    #     img_for_det = original
    img_for_det = original

    gray = cv2.cvtColor(img_for_det, cv2.COLOR_BGR2GRAY)

    # 2. Preprocessing
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    if show_steps:
        cv2.imshow("Threshold", thresh)

    # 3. Find Contours with Hierarchy
    # RETR_TREE retrieves all contours and reconstructs a full hierarchy of nested contours.
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is None:
        print("No contours found.")
        return original

    # hierarchy format: [Next, Previous, First_Child, Parent]
    hierarchy = hierarchy[0] 

    found_candidates = []

    for i, cnt in enumerate(contours):
        # Filter out noise early to save processing time
        area = cv2.contourArea(cnt)
        if area < 100: 
            continue
            
        # 4. Filter for Borders (Rectangles)
        # Approximate the contour to a polygon
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # Check if it has 4 corners (rectangle-ish) and is convex
        if len(approx) == 4 and cv2.isContourConvex(approx):
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h

            # Basic shape validity filters
            if w < 20 or h < 20: continue # Too small
            if not (0.5 < aspect_ratio < 2.0): continue # Too distorted (assuming square-ish borders)

            # 5. Check "Two Sets of Borders" Logic
            # We are looking for a rectangle that has a parent which is also a rectangle.
            # This 'cnt' would be the INNER border.
            
            parent_idx = hierarchy[i][3]
            
            if parent_idx != -1:
                parent_cnt = contours[parent_idx]
                p_peri = cv2.arcLength(parent_cnt, True)
                p_approx = cv2.approxPolyDP(parent_cnt, 0.02 * p_peri, True)

                # If parent is also a rectangle
                if len(p_approx) == 4:
                    # Check area ratio to ensure the inner box is a significant part of the outer box
                    px, py, pw, ph = cv2.boundingRect(parent_cnt)
                    area_inner = w * h
                    area_outer = pw * ph
                    
                    if 0.4 < (area_inner / area_outer) < 0.95:
                        # Add a small padding (margin) to remove the border line itself from the OCR
                        margin = int(min(w, h) * 0.1)
                        roi = gray[y+margin : y+h-margin, x+margin : x+w-margin]
                        
                        if roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
                            continue

                        if show_steps:
                            cv2.imshow("ROI", roi)

                        # 6. Recognize Text
                        # Enhance ROI for Tesseract: simple threshold
                        _, roi_thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                        cv2.imshow("ROI Threshold", roi_thresh)
                        
                        # Configuration for Tesseract:
                        # --psm 7: Treat the image as a single text line.
                        # whitelist: only digits
                        config = r'--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789'
                        
                        text = pytesseract.image_to_string(roi_thresh, config=config).strip()

                        if text:
                            # Map coordinates back to original scale for visualization
                            ox, oy, ow, oh = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
                            
                            # Draw valid detection
                            cv2.rectangle(original, (ox, oy), (ox + ow, oy + oh), (0, 255, 0), 2)
                            # Label
                            label_pos = (ox, oy - 10) if oy - 10 > 10 else (ox, oy + oh + 20)
                            cv2.putText(original, f"Val: {text}", label_pos, 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            
                            found_candidates.append(text)
                            print(f"Detected Number: {text} at [x={ox}, y={oy}]")

    return original

def run_inference(source, show_steps=False, frame_step=5):
    # Check if source is an image file
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    if isinstance(source, str) and os.path.isfile(source) and any(source.lower().endswith(ext) for ext in image_extensions):
        # Image mode
        img = cv2.imread(source)
        if img is None:
            print(f"Error: Could not read image {source}")
            return
        
        processed_img = detect_and_read_numbers(img, show_steps)
        cv2.imshow("Detections", processed_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        # Video/Webcam mode
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: Could not open video source {source}")
            return

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            # Only process every Nth frame to avoid decoder lag (HEVC POC errors)
            if frame_count % frame_step != 0:
                continue
                
            processed_frame = detect_and_read_numbers(frame, show_steps)
            
            # Resize display for high-res frames so they fit on your screen
            display_frame = processed_frame
            if processed_frame.shape[1] > 1280:
                display_scale = 1280.0 / processed_frame.shape[1]
                display_frame = cv2.resize(processed_frame, (0,0), fx=display_scale, fy=display_scale)
            
            cv2.imshow("Detections", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def generate_sample_image(filename="sample_numbers.png"):
    """Generates a test image with double borders and numbers."""
    # White background
    img = np.ones((400, 600, 3), dtype="uint8") * 255
    
    def draw_marker(cx, cy, number):
        # Outer border
        cv2.rectangle(img, (cx-60, cy-60), (cx+60, cy+60), (0,0,0), 3)
        # Inner border
        cv2.rectangle(img, (cx-40, cy-40), (cx+40, cy+40), (0,0,0), 3)
        # Number
        text_size = cv2.getTextSize(str(number), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        tx = cx - text_size[0] // 2
        ty = cy + text_size[1] // 2
        cv2.putText(img, str(number), (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 3)

    draw_marker(100, 100, 7)
    draw_marker(300, 200, 3)
    draw_marker(500, 100, 9)
    
    cv2.imwrite(filename, img)
    print(f"Generated sample image: {filename}")
    return filename

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Detect numbers inside double borders.")
    parser.add_argument("--source", type=str, help="Path to input image, video file, or camera index (e.g., '0').")
    parser.add_argument("--step", type=int, default=5, help="Process every Nth frame (default: 5).")
    args = parser.parse_args()

    if args.source:
        run_inference(args.source, show_steps=True, frame_step=args.step)
    else:
        test_file = generate_sample_image()
        run_inference(test_file, show_steps=True, frame_step=args.step)