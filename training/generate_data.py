import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import torch
from backend.model import get_depth



CLEAN_DIR = "../clean"
ATTACKED_DIR = "../attacked"

CLEAN_DIR = "data/clean_depth"
ATTACK_DIR = "data/attacked_depth"

os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(ATTACK_DIR, exist_ok=True)

def normalize(depth):
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth = (depth * 255).astype("uint8")
    return depth

files = sorted(os.listdir(CLEAN_DIR))

for file in files:
    if not file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    orig_path = os.path.join(CLEAN_DIR, file)
    att_path = os.path.join(ATTACKED_DIR, file)

    if not os.path.exists(att_path):
        print(f"Skipping {file} (no attacked match)")
        continue

    # read images
    img_orig = cv2.imread(orig_path)
    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)

    img_att = cv2.imread(att_path)
    img_att = cv2.cvtColor(img_att, cv2.COLOR_BGR2RGB)

    # generate depth
    depth_clean = get_depth(img_orig)
    depth_attack = get_depth(img_att)

    depth_clean = normalize(depth_clean)
    depth_attack = normalize(depth_attack)

    name = os.path.splitext(file)[0] + ".png"

    cv2.imwrite(os.path.join(CLEAN_DIR, name), depth_clean)
    cv2.imwrite(os.path.join(ATTACK_DIR, name), depth_attack)

    print(f"Processed {file}")

print("✅ Dataset generation complete")