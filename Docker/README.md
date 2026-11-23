## ChronoRoot 2.1 Docker

This directory contains the Dockerfile and documentation for the official ChronoRoot 2.1 Docker image. This image provides the recommended way to use all components of the ChronoRoot 2.1 platform in a consistent, isolated environment.

---

## Overview

The Docker image `ngaggion/chronoroot` contains a complete environment for running all ChronoRoot 2.1 components:

* **nnUNetv2** for AI-powered segmentation.
* **Standard Root Phenotyping Interface** for detailed individual plant analysis.
* **Screening Interface** for high-throughput batch processing.
* **Functional PCA analysis tools** (scikit-fda).
* **AI Models** are pre-configured within the environment.

The image is based on `pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime` and uses a single unified conda environment (`ChronoRoot`).

---

## System Requirements

* **Docker** installed on your system.
* **NVIDIA GPU** with appropriate drivers (required for segmentation performance).
* **NVIDIA Container Toolkit** (nvidia-docker2) installed.
* At least **30GB** of free disk space.
* **16GB RAM** recommended.

---

## Installation & Setup

### 1. Install Docker & NVIDIA Toolkit

Follow the [official Docker installation guide](https://docs.docker.com/get-docker/). For GPU acceleration, you must install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

Verify your setup with:

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi

```

### 2. Download the Image

We provide several versions of the image:

| Image Tag | Description |
| --- | --- |
| `ngaggion/chronoroot:latest` | **Standard.** Full application + Source code + Demo data. |
| `ngaggion/chronorootbase:full` | Base environment + Demo imaging data (No source code). |
| `ngaggion/chronorootbase:nodemo` | Minimal environment only (No demo data, no source code). |

```bash
docker pull ngaggion/chronoroot:latest

```

---

## Running ChronoRoot 2.1

To avoid permission issues where files created by Docker are owned by "root", we launch the container using your current User ID (`-u`). This ensures all results saved to your host machine belong to you.

### Linux Systems (Ubuntu/Debian/etc.)

```bash
# 1. Allow Docker to access your X server for the GUI
xhost +local:docker

# 2. Define your local data path
MOUNT="/path/to/your/data"

# 3. Run the container
docker run -it --gpus all \
    -u $(id -u):$(id -g) \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    -v $MOUNT:/DATA/ \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --shm-size=8gb \
    ngaggion/chronoroot:latest

# 4. (Optional) Restrict X server access when finished
xhost -local:docker

```

### Windows Systems (WSL2)

```bash
MOUNT="/mnt/c/path/to/your/data"

docker run -it --gpus all \
    -u $(id -u):$(id -g) \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    -v $MOUNT:/DATA/ \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /mnt/wslg:/mnt/wslg \
    -e DISPLAY \
    -e WAYLAND_DISPLAY \
    -e XDG_RUNTIME_DIR \
    -e PULSE_SERVER \
    --shm-size=8gb \
    ngaggion/chronoroot:latest

```

---

## Usage Inside the Container

The container includes convenient aliases for quick access:

```bash
chronoroot     # Launch Standard Root Phenotyping Interface
screening      # Launch High-throughput Screening Interface
segmentation   # Launch Segmentation Interface

```

**Manual Activation:**
If the aliases are not available in your shell:

```bash
conda activate ChronoRoot
cd /app/chronoRootApp
python run.py

```

---

## Docker Image Structure

ChronoRoot is built using a layered architecture to allow for flexible updates:

1. **`chronorootbase:nodemo`**: The core OS + CUDA + Conda environment.
2. **`chronorootbase:full`**: Inherits `nodemo` and adds the Demo imaging data.
3. **`chronoroot:latest`**: Inherits `full` and clones the [ChronoRoot2 Repository](https://github.com/ngaggion/ChronoRoot2.git) (including AI models) into `/app`.

---

## Data Management

The Docker container mounts your local directory to `/DATA/` inside the container.

* **Persistence**: All data in `/DATA/` persists on your hard drive after the container is closed.
* **Ownership**: Because we use the `-u` flag, all files created inside the container will be owned by your local user account.

---

## Troubleshooting

* **Display Issues**: If the GUI doesn't open, ensure you ran `xhost +local:docker`. On WSL2, ensure you are using a version of Windows that supports WSLg.
* **Shared Memory**: If the segmentation process crashes, increase the shared memory limit by changing `--shm-size=8gb` to `--shm-size=16gb`.

## Getting Help

1. Check the [main repository issues](https://github.com/ngaggion/ChronoRoot2/issues)
2. Open a new issue with detailed logs.