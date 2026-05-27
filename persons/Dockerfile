# Use a slim python image for a smaller, more secure footprint
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY ./requirements-cpu-only.txt .

# Force installation of CPU-only versions of Torch before other requirements
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch torchvision && \
    pip install --no-cache-dir -r requirements-cpu-only.txt

COPY persons_detection.py .

ENTRYPOINT ["python", "persons_detection.py"]