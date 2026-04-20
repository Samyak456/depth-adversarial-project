import numpy as np
import cv2

def add_noise(image: np.ndarray) -> np.ndarray:
    """Adds Gaussian noise to the image."""
    attacked_img = image.copy()
    noise = np.random.normal(0, 25, attacked_img.shape).astype(np.float32)
    noisy_img = attacked_img.astype(np.float32) + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def add_patch(image: np.ndarray) -> np.ndarray:
    """Adds a red patch to the image."""
    attacked_img = image.copy()
    h, w, _ = attacked_img.shape
    x, y = w // 3, h // 3
    # Apply red block (assuming RGB because read_image returns RGB)
    attacked_img[y:y+100, x:x+100] = [255, 0, 0]
    return attacked_img

def add_stripes(image: np.ndarray) -> np.ndarray:
    """Adds white vertical stripes to the image."""
    attacked_img = image.copy()
    for i in range(0, attacked_img.shape[1], 20):
        attacked_img[:, i:i+5] = [255, 255, 255]
    return attacked_img
