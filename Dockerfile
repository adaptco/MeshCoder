# Stage 1: Download and extract Blender
FROM ubuntu:22.04 AS blender

ARG BLENDER_VERSION="4.0.0"
ARG BLENDER_URL="https://mirror.cs.uchicago.edu/blender/release/Blender4.0/blender-4.0.0-linux-x86_64.tar.xz"

# Install wget and tar
RUN apt-get update && apt-get install -y wget tar xz-utils --no-install-recommends && rm -rf /var/lib/apt/lists/*

# Download and extract Blender
RUN wget -q -O blender.tar.xz "${BLENDER_URL}" && \
    mkdir -p /opt/blender && \
    tar -xf blender.tar.xz -C /opt/blender --strip-components=1 && \
    rm blender.tar.xz

# Stage 2: Install Python dependencies
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime AS dependencies

# Copy requirements files
COPY requirements.txt .

# Install Python packages, excluding torch since it's in the base image
RUN grep -v '^torch' requirements.txt > requirements.no-torch.txt
RUN pip install --no-cache-dir -r requirements.no-torch.txt
RUN pip install --no-cache-dir vllm pytest-mock auditnlg

# Stage 3: Final image
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime AS final

# Create a non-root user
RUN groupadd -r appuser && useradd -r -g appuser -s /bin/bash -d /app appuser
RUN mkdir /app && chown -R appuser:appuser /app

# Copy Blender from the blender stage
COPY --from=blender /opt/blender /opt/blender

# Copy installed packages from the dependencies stage
COPY --from=dependencies /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Set environment variables for Blender
ENV PATH="/opt/blender:${PATH}"
ENV PYTHONPATH="/opt/blender/4.0/python:${PYTHONPATH}"

# Copy application code
COPY . /app
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser
WORKDIR /app

# Set a default command to start an interactive bash session
CMD ["/bin/bash"]
