# persons-detection
Based on the ultralytics google collab: https://colab.research.google.com/github/eugenesiow/practical-ml/blob/master/notebooks/Detect_Persons_From_Image_YOLOv5.ipynb#scrollTo=s7QIcl4aCrso 

## Install requirements
- create virtual environment:
``python -m venv .venv``
- activate environment:
``source .venv/bin/activate``
- install requirements
``pip install -r requirements.txt``

- if needed, before running:
``export QT_QPA_PLATFORM=xcb``

- start influxdb docker container:
``sudo docker run -d -p 8086:8086   --name influxdb   -v influxdb2_data:/var/lib/influxdb2   influxdb:2.0``