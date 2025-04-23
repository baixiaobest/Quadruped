import modal

# Define custom image with MuJoCo 3.3.1
default_image = (
    modal.Image.debian_slim()
    .apt_install(
        "wget",
        "unzip",
        "patchelf",
        "libosmesa6-dev",
        "libgl1-mesa-dev",
        "libglfw3",
        "libglew-dev",
        "libxcursor-dev",
        "libxinerama-dev",
        "libegl1",
    )
    # Install MuJoCo 3.3.1
    .run_commands(
        "mkdir -p /root/.mujoco",
        "wget https://github.com/google-deepmind/mujoco/releases/download/3.3.1/mujoco-3.3.1-linux-x86_64.tar.gz -O mujoco.tar.gz",
        "tar -xf mujoco.tar.gz -C /root/.mujoco",
        "rm mujoco.tar.gz",
    )
    # Set environment variables
    .env({
        "LD_LIBRARY_PATH": "/root/.mujoco/mujoco-3.3.1/lib:/usr/lib/nvidia",
        "MUJOCO_PY_MUJOCO_PATH": "/root/.mujoco/mujoco-3.3.1",
    })
    # Install Python dependencies
    .pip_install(
        "gymnasium==1.0.0",
        "mujoco==3.3.1",  # Official Python bindings
        "stable-baselines3",
        "torch",
        "numpy",
        "imageio",
    )
    .add_local_python_source("RL")
)

IMAGES = {'default': default_image}