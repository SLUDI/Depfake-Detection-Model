"""
Configuration file for unified deepfake detection and face verification system
"""
import os

# ==================== MODEL PATHS ====================
# Deepfake Detection Model (EfficientNetB4)
DEEPFAKE_MODEL_PATH = os.getenv(
    "DEEPFAKE_MODEL_PATH", 
    "/app/model/effb4_ai_detector.pt"
)

# FaceNet Model (InceptionResnetV1)
FACENET_MODEL = "vggface2"  # Pretrained on VGGFace2 dataset

# ==================== DETECTION THRESHOLDS ====================
# Deepfake detection threshold (probability > threshold = fake)
DEEPFAKE_THRESHOLD = float(os.getenv("DEEPFAKE_THRESHOLD", "0.5"))

# Face verification threshold (similarity >= threshold = match)
VERIFICATION_THRESHOLD = float(os.getenv("VERIFICATION_THRESHOLD", "0.6"))

# High confidence deepfake threshold for early exit
DEEPFAKE_HIGH_CONFIDENCE = float(os.getenv("DEEPFAKE_HIGH_CONFIDENCE", "0.8"))

# ==================== VIDEO PROCESSING ====================
# Frame sampling interval (process every Nth frame)
FRAME_INTERVAL = int(os.getenv("FRAME_INTERVAL", "5"))

# Maximum number of faces to extract from video
MAX_FACES = int(os.getenv("MAX_FACES", "30"))

# Batch size for model inference
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))

# ==================== IMAGE SIZES ====================
# FaceNet input size
FACENET_IMAGE_SIZE = 160

# EfficientNetB4 input size
EFFICIENTNET_IMAGE_SIZE = 380

# ==================== LIVENESS DETECTION ====================
# Eye Aspect Ratio threshold for blink detection
EAR_THRESHOLD = 0.20

# Minimum consecutive frames for blink
CONSECUTIVE_FRAMES = 3

# Minimum blinks required for liveness
MIN_BLINKS_FOR_LIVENESS = 1

# ==================== FEATURE FLAGS ====================
# Enable Grad-CAM visualization for deepfake detection
ENABLE_GRADCAM = os.getenv("ENABLE_GRADCAM", "false").lower() == "true"

# Enable liveness detection (disabled by default for Docker compatibility)
ENABLE_LIVENESS = os.getenv("ENABLE_LIVENESS", "false").lower() == "true"

# Enable deepfake detection in verification flow
ENABLE_DEEPFAKE_CHECK = os.getenv("ENABLE_DEEPFAKE_CHECK", "true").lower() == "true"

# ==================== PERFORMANCE ====================
# Enable early exit on high confidence deepfake detection
ENABLE_EARLY_EXIT = os.getenv("ENABLE_EARLY_EXIT", "true").lower() == "true"

# Use GPU if available
USE_GPU = os.getenv("USE_GPU", "true").lower() == "true"
