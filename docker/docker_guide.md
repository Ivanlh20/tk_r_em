# Docker guide
Docker provides an isolated environment, known as a container, where applications can run consistently across different platforms. To help get started with docker, here is a simple guide.

github: https://github.com/Ivanlh20/tk_r_em
dockerhub: https://hub.docker.com/repository/docker/dreamleadsz/tk_r_em/general


### Prerequisites Recap
1. **For windows user: Docker with WSL 2 backend enabled**.
2. **Visual Studio Code with [Docker](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-docker) and [Remote Development](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack) extensions**
3. **CUDA GPU Support** (optional): If using the GPU (CUDA) version, ensure your system supports GPU passthrough with Docker.

### Step 1: Pull the Docker Image
In your WSL terminal (e.g., Ubuntu), pull the `dreamleadsz/tk_r_em` image:
```bash
docker pull dreamleadsz/tk_r_em:cuda
docker pull dreamleadsz/tk_r_em:cpu
```

### Step 2: Run the Docker Container

To use either the GPU (CUDA) or CPU version, choose the appropriate command and replace `path_to_local_folder` with the path to a local folder you’d like to mount to `/data` in the container.

1. **Run with CUDA (GPU Support)**:
   ```bash
   docker run -it --rm -v path_to_local_folder:/data --gpus all dreamleadsz/tk_r_em:cuda
   ```

2. **Run with CPU Only**:
   ```bash
   docker run -it --rm -v path_to_local_folder:/data dreamleadsz/tk_r_em:cpu
   ```

3. **Run with docker compse**
    ```bash
    docker compose up -d & # to run the container in the background
    docker compose down # to close the container if you do not need anymore
    ```

These commands will:
- Start a container interactively (`-it`).
- Automatically remove the container upon exit (`--rm`).
- Mount a local folder to `/data` in the container (`-v path_to_local_folder:/data`).
- Use GPU resources if available (only in the CUDA version).

### Step 3: Verify the Container is Running
Check that the container is running with:
```bash
docker ps
```

### Step 4: Attach to the Docker Container from VS Code
1. Open **Visual Studio Code**.
2. Open the **Command Palette** (`Ctrl+Shift+P`) and choose:
   ```
   > Remote-Containers: Attach to Running Container...
   ```
3. Select the running container (look for the name or ID in the list).
4. Once attached, you’ll have access to the container environment, including the `/data` folder where your local files are mounted.

### Extra: Build your own Docker image

```bash
docker login

# Build the CPU image
docker build . -f Dockerfile.cpu -t <your-dockerhub-username>/tk_r_em:cpu

# Build the GPU image
docker build . -f Dockerfile -t <your-dockerhub-username>/tk_r_em:gpu

# Push the CPU image
docker push <your-dockerhub-username>/tk_r_em:cpu

# Push the GPU image
docker push <your-dockerhub-username>/tk_r_em:gpu
```
