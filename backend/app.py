import os
import shutil
import logging
from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .model import get_depth, ensure_dependencies
from .attacks import add_noise, add_patch, add_stripes
from .utils import (
    read_image,
    save_image,
    normalize_depth_map,
    compute_error,
    create_visualization
)

app = FastAPI(title="Adversarial Depth API")
logger = logging.getLogger(__name__)

# ✅ DEFINE PATHS FIRST
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"

# ✅ THEN MOUNT
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

def ensure_folders():
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


ensure_folders()


@app.on_event("startup")
def startup_event():
    ensure_dependencies()


@app.get("/health")
def health_check():
    return {"status": "running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    ensure_folders()

    if not file.filename:
        return JSONResponse({"error": "No file uploaded"}, status_code=400)

    file_ext = os.path.splitext(file.filename)[-1].lower()
    if file_ext not in [".jpg", ".jpeg", ".png", ".bmp"]:
        print("Saved visualization at:", visual_path)
        return JSONResponse(
            {"error": f"Invalid image format: {file_ext}"},
            status_code=400
        )

    input_path = UPLOADS_DIR / file.filename

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # -------------------------
        # LOAD IMAGE
        # -------------------------
        original_img = read_image(str(input_path))

        # -------------------------
        # ATTACKS
        # -------------------------
        attacked_noise = add_noise(original_img.copy())
        attacked_patch = add_patch(original_img.copy())
        attacked_stripes = add_stripes(original_img.copy())

        base_name = os.path.splitext(file.filename)[0]

        # -------------------------
        # SAVE ORIGINAL + ATTACKS
        # -------------------------
        out_orig_path = str(OUTPUTS_DIR / f"{base_name}_original.jpg")
        out_noise_path = str(OUTPUTS_DIR / f"{base_name}_noise.jpg")
        out_patch_path = str(OUTPUTS_DIR / f"{base_name}_patch.jpg")
        out_stripes_path = str(OUTPUTS_DIR / f"{base_name}_stripes.jpg")

        save_image(out_orig_path, original_img)
        save_image(out_noise_path, attacked_noise)
        save_image(out_patch_path, attacked_patch)
        save_image(out_stripes_path, attacked_stripes)

        # -------------------------
        # DEPTH ESTIMATION
        # -------------------------
        depth_original = get_depth(original_img)
        depth_noise = get_depth(attacked_noise)
        depth_patch = get_depth(attacked_patch)
        depth_stripes = get_depth(attacked_stripes)

        # -------------------------
        # NORMALIZE DEPTH MAPS
        # -------------------------
        norm_depth_orig = normalize_depth_map(depth_original)
        norm_depth_noise = normalize_depth_map(depth_noise)
        norm_depth_patch = normalize_depth_map(depth_patch)
        norm_depth_stripes = normalize_depth_map(depth_stripes)

        out_d_orig_path = str(OUTPUTS_DIR / f"{base_name}_depth_original.png")
        out_d_noise_path = str(OUTPUTS_DIR / f"{base_name}_depth_noise.png")
        out_d_patch_path = str(OUTPUTS_DIR / f"{base_name}_depth_patch.png")
        out_d_stripes_path = str(OUTPUTS_DIR / f"{base_name}_depth_stripes.png")

        save_image(out_d_orig_path, norm_depth_orig)
        save_image(out_d_noise_path, norm_depth_noise)
        save_image(out_d_patch_path, norm_depth_patch)
        save_image(out_d_stripes_path, norm_depth_stripes)

        # -------------------------
        # METRICS
        # -------------------------
        metrics = {
            "noise_error": compute_error(depth_original, depth_noise),
            "patch_error": compute_error(depth_original, depth_patch),
            "stripes_error": compute_error(depth_original, depth_stripes),
        }

        # -------------------------
        # ADVERSARIAL DETECTION SYSTEM
        # -------------------------
        threshold = 0.15

        adversarial_detected = (
            metrics["noise_error"] > threshold or
            metrics["patch_error"] > threshold or
            metrics["stripes_error"] > threshold
        )

        warning_message = (
            "Adversarial perturbation detected: depth estimation may be unreliable."
            if adversarial_detected
            else "No significant adversarial effect detected."
        )

        # -------------------------
        # VISUALIZATION
        # -------------------------
        visual_path = str(OUTPUTS_DIR / f"{base_name}_visualization.png")

        create_visualization(
            {
                "original": original_img,
                "noise": attacked_noise,
                "patch": attacked_patch,
                "stripes": attacked_stripes,
            },
            {
                "original": depth_original,
                "noise": depth_noise,
                "patch": depth_patch,
                "stripes": depth_stripes,
            },
            visual_path
        )

        # -------------------------
        # FINAL RESPONSE
        # -------------------------
        return {
            "status": "success",
            "images": {
                "original": f"/outputs/{base_name}_original.jpg",
                "noise": f"/outputs/{base_name}_noise.jpg",
                "patch": f"/outputs/{base_name}_patch.jpg",
                "stripes": f"/outputs/{base_name}_stripes.jpg"
     },
            "depth": {
                "original": f"/outputs/{base_name}_depth_original.png",
                "noise": f"/outputs/{base_name}_depth_noise.png",
                "patch": f"/outputs/{base_name}_depth_patch.png",
                "stripes": f"/outputs/{base_name}_depth_stripes.png"
        },
            "metrics": metrics,
            "visualization": f"/outputs/{base_name}_visualization.png",
            "detection": {
                "adversarial_detected": adversarial_detected,
                "warning": warning_message,
                "threshold": threshold
            }
        }

    except Exception as e:
        logger.exception("Error processing predict request:")
        return JSONResponse({"error": str(e)}, status_code=500)