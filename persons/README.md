# Person Detection Service

A robust Python service utilizing YOLOv5m to detect persons in video streams (RTSP, Webcams, or Files) and log detection events to InfluxDB. Designed to run efficiently in Docker environments using CPU-only inference.

## Core Functionality
- **Real-time Detection:** Process RTSP streams or local camera hardware.
- **Interval Logic:** Reduce CPU load by detecting at specific intervals (e.g., every 60 seconds).
- **InfluxDB Integration:** Automatic logging of person counts and timestamps.
- **Test Mode:** Generate annotated video files to verify model performance.

## Argument Reference

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--source` | `0` | Input source: RTSP URL, local file path, or webcam index. |
| `--detection-interval` | `10` | Seconds between detection passes. Set `0` for continuous. |
| `--device` | `cpu` | Inference hardware (`cpu` or `0` for CUDA). |
| `--no-view` | `False` | Disables the GUI window. **Required for Docker/SSH.** |
| `--test-mode` | `False` | Saves the output with bounding boxes to a file. |
| `--influx-url` | - | Full URL of your InfluxDB instance. |
| `--influx-token` | - | Authentication token for InfluxDB. |

## Common Use Cases

### 1. Production Monitoring (Headless)
Run in the background, check every minute, and log to a remote database.
```bash
python persons_detection.py --source rtsp://camera_url --no-view --detection-interval 60 --influx-url http://influx-server:8086
```

### 2. Performance Verification (Test Mode)
Process a video file, save the results to disk to see what the model "sees," and skip database logging.
```bash
python persons_detection.py --source my_video.mp4 --test-mode --output-file result.mp4
```

### 3. High-Frequency Security
Continuous detection on a local webcam with a high confidence threshold.
```bash
python persons_detection.py --source 0 --detection-interval 0
```

---

## Docker Deployment

### Using Docker Compose (Recommended)
The service is pre-configured to work with `docker-compose`. Note that `network_mode: host` is used to ensure the container can reach RTSP streams on your local network.

1. **Configure Environment:** Update the `command` section in `docker-compose.yml` with your RTSP credentials and InfluxDB tokens.
2. **Launch:**
   ```bash
   docker-compose up -d --build
   ```
3. **Monitor Logs:**
   ```bash
   docker-compose logs -f person-detector
   ```

### Manual Docker Commands
**Build the image:**
```bash
docker build -t persons-detection:latest .
```

**Run the container:**
```bash
docker run -d \
  --name person-detector \
  --network host \
  -e PYTHONUNBUFFERED=1 \
  persons-detection:latest \
  --source rtsp://your-camera-url \
  --no-view \
  --influx-url http://your-influx-ip:8086 \
  --influx-token YOUR_INFLUX_TOKEN \
  --influx-org YOUR_ORG \
  --influx-bucket YOUR_BUCKET \
  --detection-interval 60
```
**Run with webcam**
```bash
sudo docker run -it --device=/dev/video0:/dev/video0 persons-detection:1.0 --source 0 --detection-interval 0 --no-view
```
**Run with video file**
```bash
sudo docker run -it -v your/folder/with/videos:/media/data persons-detection:1.0 --source /media/data/your_video.mp4 --detection-interval 0 --test-mode --no-view --output-file /media/data/output_test.mp4
```

## Technical Notes
- **Base Image:** Uses `python:3.12-slim` for a minimal footprint.
- **Dependencies:** Uses `opencv-python-headless` to avoid requirements for X11/GUI libraries inside the container.
- **Networking:** If your RTSP stream is on the local LAN, the container **must** use `--network host` or be on the same Docker bridge as the stream producer.
- **Caching:** The YOLOv5 weights (`yolov5m.pt`) are downloaded on the first run. To persist these, mount a volume to `/root/.cache/torch`.

## Troubleshooting
- **EOFError:** Ensure `--no-view` is passed when running in Docker; otherwise, Torch Hub will try to prompt for confirmation in a non-interactive shell.
- **No Detections:** Verify that `ffmpeg` is installed (included in the provided Dockerfile) and that the RTSP URL is reachable from the host.

## Other commands and issues
- if needed, before running:
``export QT_QPA_PLATFORM=xcb``

- start influxdb docker container:
``sudo docker run -d -p 8086:8086   --name influxdb   -v influxdb2_data:/var/lib/influxdb2   influxdb:2.0``

### Manual Push
0. `docker login ghcr.io` # Example for GitHub Container Registry
1. `docker build -t persons-detection:latest .`
2. `docker tag persons-detection:latest <registry_url>/<repo>:latest`
3. `docker push <registry_url>/<repo>:latest`