# persons-detection
Based on the ultralytics google collab: https://colab.research.google.com/github/eugenesiow/practical-ml/blob/master/notebooks/Detect_Persons_From_Image_YOLOv5.ipynb#scrollTo=s7QIcl4aCrso 

## Install requirements
- please read the main README.md file

## Before running the script
- if needed, before running:
``export QT_QPA_PLATFORM=xcb``

- start influxdb docker container:
``sudo docker run -d -p 8086:8086   --name influxdb   -v influxdb2_data:/var/lib/influxdb2   influxdb:2.0``

## Run script: 
- python persons_detection.py # with optional parameters 