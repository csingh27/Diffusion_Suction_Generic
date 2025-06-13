FROM songhesd/cuda:10.0-cudnn7-devel-ubuntu16.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Basic dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    software-properties-common \
    libssl-dev \
    libffi-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    liblzma-dev \
    zlib1g-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libgdbm-dev \
    libnss3-dev \
    libtk8.6 \
    libx11-dev \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender-dev

# Install Python 3.6 manually from source
WORKDIR /opt
RUN wget https://www.python.org/ftp/python/3.6.15/Python-3.6.15.tgz && \
    tar xzf Python-3.6.15.tgz && \
    cd Python-3.6.15 && \
    ./configure --enable-optimizations && \
    make -j"$(nproc)" && \
    make altinstall

# Set python3.6 as default
RUN ln -s /usr/local/bin/python3.6 /usr/bin/python && \
    ln -s /usr/local/bin/pip3.6 /usr/bin/pip

# Upgrade pip
RUN pip install --upgrade pip

# Clone your repo
RUN git clone https://github.com/TAO-TAO-TAO-TAO-TAO/Diffusion_Suction.git
WORKDIR /Diffusion_Suction

# Install Python packages
RUN pip install \
    torch==1.1.0 \
    torchvision==0.3.0 \
    matplotlib \
    opencv-python \
    plyfile \
    trimesh==2.35.39 \
    networkx==2.2 \
    scikit-learn \
    h5py \
    nibabel

# Build CUDA extensions (skip failure if install is not implemented)
WORKDIR /Diffusion_Suction/train/Sparepart
RUN python train.py install || true

CMD ["python", "train.py"]
