# Docker Setup for MeshCoder

This document outlines the Docker setup for the MeshCoder project. It provides a containerized environment with all the necessary dependencies, including Python, PyTorch, CUDA, and Blender.

## Deliverables

-   **`Dockerfile`**: A multi-stage Dockerfile that builds a production-ready image with all dependencies.
-   **`docker-compose.yml`**: A Docker Compose file for easy management of the container, including volume mounts for development and GPU support.
-   **`.env.example`**: An example environment file for managing secrets and configuration, such as the Hugging Face Hub token.

## Dockerfile Explained

The `Dockerfile` uses a multi-stage build to create a clean and efficient final image.

-   **`blender` stage**: This stage uses a lightweight Ubuntu image to download and extract the specified version of Blender. This keeps the final image clean from build-time dependencies like `wget`.
-   **`dependencies` stage**: This stage starts from an official NVIDIA PyTorch image, which includes CUDA and cuDNN. It then installs the Python dependencies from `requirements.txt`, leveraging Docker's layer caching.
-   **`final` stage**: The final image is built on the same PyTorch base image. It copies Blender and the Python packages from the previous stages. It sets up a non-root user for security, copies the application code, and sets environment variables to make Blender's Python module (`bpy`) available.

## Docker Compose Explained

The `docker-compose.yml` file provides a convenient way to manage the container.

-   **`build`**: It specifies that the container should be built using the `Dockerfile` in the current directory.
-   **`deploy`**: This section is configured to provide the container with access to all available NVIDIA GPUs, which is necessary for running the models.
-   **`volumes`**: It mounts the `src`, `recipes`, and `scripts` directories from the host into the container. This allows you to edit the code on your local machine and see the changes reflected inside the container without rebuilding the image.
-   **`mem_reservation` and `mem_limit`**: These settings manage the container's memory usage.
-   **`stdin_open` and `tty`**: These keep the container running and allow you to attach an interactive terminal.

## How to Use

1.  **Install Docker and NVIDIA Container Toolkit**: Ensure you have Docker and the NVIDIA Container Toolkit installed on your system to enable GPU support in Docker.
2.  **Create a `.env` file**: Copy the `.env.example` file to a new file named `.env` and add your Hugging Face Hub token.
    ```bash
    cp .env.example .env
    ```
3.  **Build and run the container**: Use Docker Compose to build and run the container in detached mode.
    ```bash
    docker-compose up --build -d
    ```
4.  **Access the container**: You can now access a bash shell inside the running container.
    ```bash
    docker-compose exec meshcoder /bin/bash
    ```
    From this shell, you can run any of the project's Python scripts.

## Key Best Practices Applied

-   **Multi-stage builds**: To keep the final image small and clean.
-   **Layer caching**: For faster builds by separating dependencies from source code.
-   **Non-root user**: The container runs with a non-root user to improve security.
-   **Official base images**: Using official images for PyTorch ensures a stable and well-configured environment.
-   **Development-ready**: Volume mounts for source code allow for an efficient development workflow.
-   **GPU support**: Correctly configured for GPU-accelerated tasks.
-e   **Configuration management**: Using a `.env` file for secrets.
