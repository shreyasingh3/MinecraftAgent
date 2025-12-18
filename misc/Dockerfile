FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    openjdk-8-jdk \
    xvfb \
    x11vnc \
    fluxbox \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up Python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Install Python dependencies
RUN pip install --no-cache-dir MineStudio torch psutil

# Set working directory
WORKDIR /workspace

# Copy MCU benchmark code
COPY . /workspace/

# Set environment variables for headless operation
ENV DISPLAY=:99
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64

CMD ["/bin/bash"]

