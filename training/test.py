import cv2
import torch
import matplotlib.pyplot as plt

from model import DepthCorrectionCNN

device = "cpu"

# load model
model = DepthCorrectionCNN().to(device)
model.load_state_dict(torch.load("weights/depth_correction.pth"))
model.eval()

# load test image
img = cv2.imread("data/attacked_depth/00001.jpg", 0)
img = cv2.resize(img, (256, 256))

input_img = torch.tensor(img / 255.0).unsqueeze(0).unsqueeze(0).float()

# prediction
with torch.no_grad():
    output = model(input_img).squeeze().numpy()

# plot
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.title("Attacked Depth")
plt.imshow(img, cmap="plasma")
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Corrected Depth")
plt.imshow(output, cmap="plasma")
plt.axis("off")

plt.tight_layout()
plt.show()