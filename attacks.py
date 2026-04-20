import cv2
import numpy as np

def add_noise(image):
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    return cv2.add(image, noise)

def add_patch(image):
    h, w, _ = image.shape
    x, y = w//3, h//3
    image[y:y+100, x:x+100] = [255, 0, 0]
    return image

def add_stripes(image):
    for i in range(0, image.shape[1], 20):
        image[:, i:i+5] = 255
    return image