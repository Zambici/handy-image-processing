# Simple Object Recognition Training Algorithm

This directory contains scripts to generate datasets from video footage and train custom object detection models using YOLOv8.

## Files

### `create_dataset_from_video.py`
Helper script to build a dataset from video files.
- **Functionality**: Extracts frames from videos in a specified folder.
- **Features**:
  - Supports `.mp4`, `.avi`, `.mov`, `.mkv`.
  - Implements `detect_object_lightness` to identify objects based on contrast (dark object on light background), which aids in generating masks or bounding boxes for the dataset.

### `train_object_detection.py`
Script to train a YOLOv8 model on your custom dataset.
- **Functionality**: Loads a pre-trained YOLOv8 model (e.g., `yolov8x.pt`) and fine-tunes it.
- **Requirements**: Expects a `data.yaml` file in the dataset directory to define class names and paths.

## Usage

1. **Prepare Dataset**: Use `create_dataset_from_video.py` to extract frames. You may need to modify the script to point to your input video folder.
2. **Train**: Run `train_object_detection.py` to start training. Ensure your dataset directory structure matches what YOLOv8 expects and that `data.yaml` is present.