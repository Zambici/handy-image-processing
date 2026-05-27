# persons-detection
Based on the ultralytics google collab: https://colab.research.google.com/github/eugenesiow/practical-ml/blob/master/notebooks/Detect_Persons_From_Image_YOLOv5.ipynb#scrollTo=s7QIcl4aCrso 

## Install requirements
- create virtual environment:
``python -m venv .venv``
- activate environment:
``source .venv/bin/activate``
- install requirements
``pip install -r requirements.txt``

## Quick commands
### Generate a RTSP stream from webcam
Link: https://stackoverflow.com/questions/33800086/using-ffmpeg-to-generate-rtsp-from-webcam
Commands:
``rtspServer=192.168.1.227:rtsp://192.168.1.227:8554/webCamStream``
``sudo ffmpeg -f v4l2 -framerate 30 -video_size 480x480 -i /dev/video0 -f rtsp -rtsp_transport tcp rtsp://192.168.1.227:8554/webCamStream``
``ffplay "rtsp://192.168.1.227:8554/webCamStream"``