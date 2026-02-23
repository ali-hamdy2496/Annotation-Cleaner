FROM nvidia/cuda:13.0.1-devel-ubuntu22.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    git \
    libjpeg-dev \
    libpng-dev \
    wget


RUN apt-get update && apt-get install -y python3 python3-pip

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

COPY requirements.txt /app/requirements.txt

RUN pip install -r /app/requirements.txt

