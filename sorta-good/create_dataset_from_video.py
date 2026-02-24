import cv2
import os
import numpy as np
import sys
import glob
import shutil
import random

def extract_frames(input_folder, output_base_dir="dataset", frame_step=5):
    """
    Extracts frames from all videos in a folder and saves them to a directory.

    Args:
        input_folder (str): Path to the folder containing video files.
        output_base_dir (str): Base directory where extracted frames folder will be created.
        frame_step (int): Save every Nth frame to avoid duplicate data.
    """
    folder_name = os.path.basename(os.path.normpath(input_folder))
    frames_dir = os.path.join(output_base_dir, folder_name)
    if os.path.exists(frames_dir) and len(glob.glob(os.path.join(frames_dir, "*.jpg"))) > 0:
        print(f"WARNING: Frames directory '{frames_dir}' already exists and contains images. Skipping extraction.")
        return frames_dir

    os.makedirs(frames_dir, exist_ok=True)

    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(input_folder, ext)))
    
    video_files.sort()

    if not video_files:
        print(f"No video files found in {input_folder}")
        return frames_dir

    print(f"Extracting frames from {len(video_files)} videos in {input_folder} to {frames_dir}...")

    global_saved_count = 0

    for video_path in video_files:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            continue
        
        print(f"Processing {os.path.basename(video_path)}...")
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_step == 0:
                frame_filename = os.path.join(frames_dir, f"frame_{global_saved_count:05d}.jpg")
                cv2.imwrite(frame_filename, frame)
                global_saved_count += 1

            frame_count += 1

        cap.release()
    
    print(f"Done! Extracted {global_saved_count} frames.")
    return frames_dir

def detect_object_lightness(frame, bg_image):
    """
    Detects object using lightness thresholding (dark object on light background).
    Returns:
        bbox: (x, y, w, h) or None
        mask: binary mask used for detection
        debug_layer: specific channel used for detection (for visualization)
    """
    # --- Determine dynamic threshold from the white background image ---
    bg_lab = cv2.cvtColor(bg_image, cv2.COLOR_BGR2LAB)
    l_channel_bg, _, _ = cv2.split(bg_lab)
    # Use the median as it's robust to outliers (e.g., a spec of dust)
    median_bg_lightness = np.median(l_channel_bg)
    # Set threshold lower than the background lightness. L* values are 0-255.
    threshold = int(median_bg_lightness * 0.6) # e.g., 90% of median lightness

    # Convert frame to LAB color space, where L* is the lightness channel
    lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, _, _ = cv2.split(lab_frame)
    
    # Create a mask where pixels darker than the threshold are selected.
    # THRESH_BINARY_INV makes pixels < threshold white (255), and others black (0).
    _, mask = cv2.threshold(l_channel, threshold, 255, cv2.THRESH_BINARY_INV)
    
    # --- Morphological Operations to clean the mask ---
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # Remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # Fill holes
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 1000:
            return cv2.boundingRect(largest_contour), mask, l_channel
            
    return None, mask, l_channel

def process_frames(frames_dir, background_path, output_base_dir, class_id=0, debug=True):
    """
    Reads frames from a directory, subtracts background to isolate objects, and generates YOLO dataset.
    """
    if os.path.exists(os.path.join(output_base_dir, 'train')) or \
       (os.path.exists(os.path.join(output_base_dir, 'images')) and len(glob.glob(os.path.join(output_base_dir, 'images', '*.jpg'))) > 0):
        print(f"WARNING: Output directory '{output_base_dir}' already contains processed data. Skipping processing.")
        return

    bg_image = cv2.imread(background_path)
    if bg_image is None:
        print(f"Error: Could not load background image from {background_path}")
        return

    images_dir = os.path.join(output_base_dir, 'images')
    labels_dir = os.path.join(output_base_dir, 'labels')
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    print(f"Processing {len(frame_files)} frames from {frames_dir}...")
    
    if debug:
        print("Debug mode ON: Press SPACEBAR to advance, 'd' to discard current, 'q' to quit.")

    processed_count = 0

    for frame_path in frame_files:
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        # Call the detection function
        bbox, mask, l_channel = detect_object_lightness(frame, bg_image)

        if bbox:
            x, y, box_w, box_h = bbox
            h, w = frame.shape[:2]

            # Normalize coordinates
            x_center = (x + box_w / 2) / w
            y_center = (y + box_h / 2) / h
            norm_w = box_w / w
            norm_h = box_h / h

            base_filename = os.path.splitext(os.path.basename(frame_path))[0]
            img_out_path = os.path.join(images_dir, base_filename + ".jpg")
            txt_out_path = os.path.join(labels_dir, base_filename + ".txt")

            cv2.imwrite(img_out_path, frame)

            with open(txt_out_path, "w") as f:
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

            processed_count += 1

            if debug:
                vis_frame = frame.copy()
                cv2.rectangle(vis_frame, (x, y), (x + box_w, y + box_h), (0, 255, 0), 2)
                # Show the lightness channel and the resulting mask for debugging
                cv2.imshow("L-Channel", l_channel)
                cv2.imshow("Mask", mask)
                cv2.imshow("Preview", vis_frame)
                
                # Wait for spacebar to continue
                quit_loop = False
                while True:
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord(' '):
                        break
                    elif key == ord('d'):
                        print(f"Discarding {os.path.basename(img_out_path)}...")
                        if os.path.exists(img_out_path):
                            os.remove(img_out_path)
                        if os.path.exists(txt_out_path):
                            os.remove(txt_out_path)
                        processed_count -= 1
                        break
                    elif key == ord('q'):
                        quit_loop = True
                        break
                
                if quit_loop:
                    break

    cv2.destroyAllWindows()
    print(f"Generated {processed_count} labeled images in '{output_base_dir}'.")

def split_dataset(base_dir, split_ratio=0.8):
    """
    Splits the generated dataset into train and val sets.
    """
    if os.path.exists(os.path.join(base_dir, 'train')) and os.path.exists(os.path.join(base_dir, 'val')):
        print(f"WARNING: Train/Val directories already exist in '{base_dir}'. Skipping split.")
        return

    print(f"Splitting dataset in {base_dir} into train/val...")
    
    src_images = os.path.join(base_dir, "images")
    src_labels = os.path.join(base_dir, "labels")

    if not os.path.exists(src_images):
        print("No images directory found to split.")
        return

    # Create train/val structure
    for split in ['train', 'val']:
        os.makedirs(os.path.join(base_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, split, 'labels'), exist_ok=True)

    image_files = glob.glob(os.path.join(src_images, "*.jpg"))
    random.shuffle(image_files)

    split_idx = int(len(image_files) * split_ratio)
    splits = {'train': image_files[:split_idx], 'val': image_files[split_idx:]}

    for split, files in splits.items():
        for img_path in files:
            shutil.move(img_path, os.path.join(base_dir, split, 'images'))
            
            basename = os.path.splitext(os.path.basename(img_path))[0]
            txt_path = os.path.join(src_labels, basename + ".txt")
            if os.path.exists(txt_path):
                shutil.move(txt_path, os.path.join(base_dir, split, 'labels'))

    # Cleanup empty source dirs
    if os.path.exists(src_images) and not os.listdir(src_images):
        os.rmdir(src_images)
    if os.path.exists(src_labels) and not os.listdir(src_labels):
        os.rmdir(src_labels)
    
    print(f"Split complete. Train: {len(splits['train'])}, Val: {len(splits['val'])}")

def create_yaml(base_dir, class_names):
    yaml_path = os.path.join(base_dir, "data.yaml")
    if os.path.exists(yaml_path):
        print(f"WARNING: '{yaml_path}' already exists. Skipping YAML creation.")
        return

    yaml_content = f"path: {os.path.abspath(base_dir)}\ntrain: train/images\nval: val/images\n\nnames:\n"
    for i, name in enumerate(class_names):
        yaml_content += f"  {i}: {name}\n"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"Created data.yaml at {yaml_path}")

if __name__ == "__main__":
    input_folder = "../../raw_data/pumpkin"
    if len(sys.argv) >= 2:
        input_folder = sys.argv[1]

    if not os.path.isdir(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
    else:
        # Find background image (jpg, png, etc.)
        bg_files = glob.glob(os.path.join(input_folder, "*.jpg")) + \
                   glob.glob(os.path.join(input_folder, "*.png")) + \
                   glob.glob(os.path.join(input_folder, "*.jpeg"))
        
        if not bg_files:
            print(f"Error: No background image found in {input_folder}. Please ensure a background image exists.")
        else:
            background_image_path = bg_files[0]
            print(f"Using background image: {background_image_path}")

            # 1. Extract
            frames_dir = extract_frames(input_folder)
            
            # Determine dataset name and path
            dataset_name = os.path.basename(os.path.normpath(input_folder))
            output_dataset_dir = os.path.join("datasets", f"dataset_{dataset_name}")

            # 2. Process
            process_frames(frames_dir, background_image_path, output_base_dir=output_dataset_dir, debug=True)
            
            # 3. Split & Config
            split_dataset(output_dataset_dir)
            create_yaml(output_dataset_dir, [dataset_name])
