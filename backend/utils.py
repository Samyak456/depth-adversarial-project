import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# IMAGE FUNCTIONS
# -------------------------

def read_image(file_path: str) -> np.ndarray:
    """Reads an image from a path and returns it as a numpy RGB array."""
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError(f"Could not read image from {file_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def save_image(file_path: str, image: np.ndarray):
    """Saves an RGB or grayscale numpy array image to a path."""
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_path, image)


# -------------------------
# DEPTH PROCESSING
# -------------------------

def normalize_depth_map(depth_map: np.ndarray) -> np.ndarray:
    """Normalizes a depth map to 0-255 uint8 for saving."""
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    
    if depth_max - depth_min > 0:
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    
    depth_map = (depth_map * 255.0).astype(np.uint8)
    return depth_map


# -------------------------
# METRICS (NEW)
# -------------------------

def compute_error(depth1: np.ndarray, depth2: np.ndarray) -> float:
    """
    Computes Mean Absolute Error between two depth maps.
    """
    return float(np.mean(np.abs(depth1 - depth2)))


# -------------------------
# VISUALIZATION (NEW)
# -------------------------

def create_visualization(images_dict, depth_dict, save_path):
    """
    Creates a 2x4 grid visualization and saves it.
    """

    plt.figure(figsize=(12, 8))

    titles = [
        "Original", "Noise", "Patch", "Stripes",
        "Depth Original", "Depth Noise", "Depth Patch", "Depth Stripes"
    ]

    imgs = [
        images_dict["original"],
        images_dict["noise"],
        images_dict["patch"],
        images_dict["stripes"],
        depth_dict["original"],
        depth_dict["noise"],
        depth_dict["patch"],
        depth_dict["stripes"],
    ]

    for i in range(8):
        plt.subplot(2, 4, i + 1)
        plt.title(titles[i])

        if i < 4:
            plt.imshow(imgs[i])
        else:
            plt.imshow(imgs[i], cmap="plasma")

        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()