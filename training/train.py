import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from model import DepthCorrectionCNN
print("🚀 Training script started")

device = "cpu"

model = DepthCorrectionCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

clean_dir = "./data/clean_depth"
attacked_dir = "./data/attacked_depth"

files = os.listdir(clean_dir)

def load_image(path):
    img = cv2.imread(path, 0)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    return torch.tensor(img).unsqueeze(0).unsqueeze(0).float()

files = os.listdir(clean_dir)
print("Files found:", len(files))

for epoch in range(20):
    total_loss = 0

    for file in files:
        clean = load_image(os.path.join(clean_dir, file)).to(device)
        attacked = load_image(os.path.join(attacked_dir, file)).to(device)

        pred = model(attacked)
        loss = loss_fn(pred, clean)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/20, Loss: {total_loss:.4f}")

# save model
os.makedirs("weights", exist_ok=True)
torch.save(model.state_dict(), "weights/depth_correction.pth")

print("✅ Training complete")