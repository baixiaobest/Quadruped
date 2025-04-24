import modal
import sys
import os

# Add the parent directory (root) to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from RL.modal_training.image_definitions import IMAGES

image = IMAGES["default"].add_local_python_source("RL")

App = modal.App("ppo_walker", image=image)

@App.function(
    timeout=12*60*60
)
def main():
    import numpy as np
    print("Numpy version:", np.__version__)

if __name__ == "__main__":
    main.remote()