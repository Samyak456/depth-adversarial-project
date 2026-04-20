import os
import sys
import subprocess
import logging
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def ensure_dependencies():
    """Auto-installs missing torch/torchvision/timm if not present."""
    missing = []
    try:
        import torch
    except ImportError:
        missing.extend(["torch", "torchvision"])
    
    try:
        import timm
    except ImportError:
        missing.append("timm")
        
    if missing:
        logger.info(f"Missing dependencies: {missing}. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
        logger.info("Dependencies installed.")

# Check for dependencies once upon importing model.py
ensure_dependencies()

import torch

class MiDaSModel:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        # We enforce CPU-only usage here as per requirement
        self.device = torch.device("cpu")
        self.midas = None
        self.transform = None
        
        logger.info("Loading MiDaS_small model from torch.hub...")
        
        # Retry mechanism for model loading
        retries = 1
        for attempt in range(retries + 1):
            try:
                self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", force_reload=False)
                midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
                self.transform = midas_transforms.small_transform
                break
            except Exception as e:
                logger.error(f"Failed to load model on attempt {attempt+1}: {e}")
                if attempt == retries:
                    raise RuntimeError("Failed to load MiDaS model after retrying.")
                
        self.midas.to(self.device)
        self.midas.eval()
        logger.info("MiDaS model loaded successfully.")

    def get_depth(self, image: np.ndarray) -> np.ndarray:
        # MiDaS small_transform expects RGB image
        img_transformed = self.transform(image).to(self.device)
        
        with torch.no_grad():
            prediction = self.midas(img_transformed)
            
            # Predict the depth layout for original image resolution
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        output = prediction.cpu().numpy()
        return output

# Singleton so the model is only loaded once globally
_model_instance = None

def get_depth(image: np.ndarray) -> np.ndarray:
    global _model_instance
    if _model_instance is None:
        _model_instance = MiDaSModel.get_instance()
    return _model_instance.get_depth(image)
