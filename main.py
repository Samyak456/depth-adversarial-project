import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Import attacks
from attacks import add_noise, add_patch, add_stripes

# -------------------------
# LOAD MODEL (MiDaS)
# -------------------------
model_type = "MiDaS_small"

midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.eval()

transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.small_transform

# -------------------------
# LOAD IMAGE
# -------------------------
img = cv2.imread("test.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# -------------------------
# CREATE ATTACKED IMAGES
# -------------------------
attacked_noise = add_noise(img.copy())
attacked_patch = add_patch(img.copy())
attacked_stripes = add_stripes(img.copy())

# -------------------------
# DEPTH FUNCTION
# -------------------------
def get_depth(image):
    input_batch = transform(image)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    return prediction.cpu().numpy()

# -------------------------
# GET DEPTH MAPS
# -------------------------
depth_original = get_depth(img)
depth_noise = get_depth(attacked_noise)
depth_patch = get_depth(attacked_patch)
depth_stripes = get_depth(attacked_stripes)

# -------------------------
# SAVE OUTPUTS
# -------------------------
import os
os.makedirs("outputs", exist_ok=True)

cv2.imwrite("outputs/original.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
cv2.imwrite("outputs/noise.jpg", cv2.cvtColor(attacked_noise, cv2.COLOR_RGB2BGR))
cv2.imwrite("outputs/patch.jpg", cv2.cvtColor(attacked_patch, cv2.COLOR_RGB2BGR))
cv2.imwrite("outputs/stripes.jpg", cv2.cvtColor(attacked_stripes, cv2.COLOR_RGB2BGR))

cv2.imwrite("outputs/depth_original.png", depth_original)
cv2.imwrite("outputs/depth_noise.png", depth_noise)
cv2.imwrite("outputs/depth_patch.png", depth_patch)
cv2.imwrite("outputs/depth_stripes.png", depth_stripes)

# -------------------------
# VISUALIZATION
# -------------------------
plt.figure(figsize=(12,8))

titles = [
    "Original", "Noise", "Patch", "Stripes",
    "Depth Original", "Depth Noise", "Depth Patch", "Depth Stripes"
]

images = [
    img, attacked_noise, attacked_patch, attacked_stripes,
    depth_original, depth_noise, depth_patch, depth_stripes
]

for i in range(8):
    plt.subplot(2,4,i+1)
    plt.title(titles[i])
    
    if i < 4:
        plt.imshow(images[i])
    else:
        plt.imshow(images[i], cmap="plasma")
    
    plt.axis("off")

plt.tight_layout()
plt.show()