# tensorboard_server.py
# Run this script by modal deploy tensorboard_server.py,
# then visit the modal.com to access the link to the tensorboard server.
import modal
import subprocess

tensorboard_image = modal.Image.debian_slim().pip_install(
    "tensorboard==2.19.0",
    "tensorflow==2.19.0"
)

# Initialize Modal app properly
app = modal.App("tensorboard-server")  # MUST be lowercase 'app'

# Create volume
volume = modal.Volume.from_name("log")

@app.function(
    image=tensorboard_image,
    volumes={"/log": volume},
    min_containers=1,
)
@modal.web_server(
    port=8000,
    startup_timeout=180,
)
def tensorboard_server():
    command = [
        "python", "-m", "tensorboard.main",
        "--logdir", "/log/tensorboard",
        "--port", "8000",
        "--host", "0.0.0.0"
    ]
    return subprocess.Popen(command)