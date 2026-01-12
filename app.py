import os
import io
import base64
import time
import json
import logging
import tempfile
from typing import Optional, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

# Grad-CAM library for visualization
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Hugging Face Hub for model download
from huggingface_hub import hf_hub_download

# FaceNet for face verification
from facenet_pytorch import InceptionResnetV1

# Import unified system components
import config
from video_processor import OptimizedVideoProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Unified Deepfake Detection & Face Verification API",
    description="Combined deepfake detection (EfficientNetB4) and face verification (FaceNet)",
    version="3.0.0"
)

# Add CORS middleware for web/mobile app integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODEL DEFINITION ---
# Note: This model uses timm library's EfficientNetB4 architecture
# to match the checkpoint format
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("⚠️  timm library not available, will try direct checkpoint loading")

# Load the model directly from checkpoint
# The checkpoint contains a complete timm EfficientNetB4 model

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_LOCAL_PATH = os.getenv("MODEL_PATH", "/app/model/effb4_ai_detector.pt")
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO")  # Optional: HuggingFace repo for model
HF_TOKEN = os.getenv("HF_TOKEN")  # Optional: HF token for private repos

# Download model from HuggingFace Hub if configured
if HF_MODEL_REPO and not os.path.exists(MODEL_LOCAL_PATH):
    os.makedirs(os.path.dirname(MODEL_LOCAL_PATH), exist_ok=True)
    print(f"📥 Downloading model from HuggingFace Hub: {HF_MODEL_REPO}")
    hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=os.path.basename(MODEL_LOCAL_PATH),
        local_dir=os.path.dirname(MODEL_LOCAL_PATH),
        use_auth_token=HF_TOKEN
    )

# --- LOAD MODEL ---
print(f"🔧 Loading model from: {MODEL_LOCAL_PATH}")
print(f"🖥️  Using device: {DEVICE}")

if not os.path.exists(MODEL_LOCAL_PATH):
    raise FileNotFoundError(
        f"Model not found at {MODEL_LOCAL_PATH}. "
        f"Please add the model file or set HF_MODEL_REPO environment variable."
    )

# Load checkpoint
try:
    checkpoint = torch.load(MODEL_LOCAL_PATH, map_location=DEVICE)
    
    # Check if timm is available
    if not TIMM_AVAILABLE:
        raise ImportError("timm library is required but not installed. Install with: pip install timm")
    
    # Create timm EfficientNetB4 model
    print("📦 Creating EfficientNetB4 model with timm...")
    model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=1)
    
    # Load the checkpoint weights
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Load state dict
    model.load_state_dict(state_dict, strict=False)
    model = model.to(DEVICE)
    
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise


model.eval()

# --- LOAD FACENET MODEL FOR FACE VERIFICATION ---
logger.info("📦 Loading FaceNet model for face verification...")
try:
    facenet_model = InceptionResnetV1(pretrained=config.FACENET_MODEL, device=DEVICE).eval()
    logger.info("✅ FaceNet (InceptionResnetV1) loaded successfully!")
except Exception as e:
    logger.warning(f"⚠️ FaceNet model loading failed: {e}")
    logger.warning("Face verification endpoints will be disabled")
    facenet_model = None

# --- INITIALIZE VIDEO PROCESSOR ---
logger.info("🎥 Initializing optimized video processor...")
try:
    video_processor = OptimizedVideoProcessor(device=DEVICE)
    logger.info("✅ Video processor initialized!")
except Exception as e:
    logger.warning(f"⚠️ Video processor initialization failed: {e}")
    logger.warning("Video-based verification will be disabled")
    video_processor = None

# Initialize Grad-CAM for visualization (optional, can be disabled for speed)
ENABLE_GRADCAM = os.getenv("ENABLE_GRADCAM", "true").lower() == "true"
cam = None

if ENABLE_GRADCAM:
    try:
        # For timm EfficientNet, target the last convolutional layer before classifier
        # timm EfficientNet structure: blocks -> conv_head -> global_pool -> classifier
        target_layers = [model.conv_head]
        cam = GradCAM(model=model, target_layers=target_layers)
        logger.info("✅ Grad-CAM visualization enabled")
    except Exception as e:
        logger.warning(f"⚠️ Grad-CAM initialization failed: {e}")
        ENABLE_GRADCAM = False

# --- IMAGE PREPROCESSING ---
preprocess = transforms.Compose([
    transforms.Resize((380, 380)),  # EfficientNetB4 input size
    transforms.CenterCrop(380),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def tensor_from_pil(image: Image.Image) -> torch.Tensor:
    """Convert PIL image to tensor"""
    return preprocess(image).unsqueeze(0).to(DEVICE)

def to_base64_png(np_image_rgb: np.ndarray) -> str:
    """Convert numpy array to base64 PNG string"""
    _, im_buf_arr = cv2.imencode('.png', cv2.cvtColor(np_image_rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(im_buf_arr.tobytes()).decode('utf-8')

# --- API ENDPOINTS ---

@app.get("/")
async def root():
    """Root endpoint with complete API information"""
    return {
        "status": "healthy",
        "service": "Unified Deepfake Detection & Face Verification API",
        "version": "3.0.0",
        "models": {
            "deepfake_detection": "EfficientNetB4",
            "face_verification": "FaceNet (InceptionResnetV1)" if facenet_model else "Disabled",
            "liveness_detection": "MediaPipe Facial Landmarks" if config.ENABLE_LIVENESS else "Disabled"
        },
        "device": str(DEVICE),
        "endpoints": {
            "health": "/health",
            "deepfake_detection": "/api/v1/detect/deepfake",
            "legacy_predict": "/predict",
            "liveness_check": "/api/v1/verify/liveness",
            "extract_embedding": "/extract",
            "verify_with_embedding": "/verify-with-embedding",
            "verify_two_videos": "/verify-two-videos"
        },
        "features": {
            "gradcam_visualization": ENABLE_GRADCAM,
            "liveness_detection": config.ENABLE_LIVENESS,
            "deepfake_check": config.ENABLE_DEEPFAKE_CHECK,
            "early_exit_optimization": True
        },
        "recommended_thresholds": {
            "deepfake": config.DEEPFAKE_THRESHOLD,
            "face_verification": config.VERIFICATION_THRESHOLD
        },
        "usage": "POST requests to endpoints. See /docs for interactive API documentation."
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    return {
        "status": "healthy",
        "models": {
            "deepfake_detector": {
                "loaded": model is not None,
                "type": "EfficientNetB4",
                "device": str(DEVICE)
            },
            "face_verifier": {
                "loaded": facenet_model is not None,
                "type": "FaceNet (InceptionResnetV1)",
                "embedding_size": 512
            },
            "video_processor": {
                "loaded": video_processor is not None,
                "liveness_enabled": config.ENABLE_LIVENESS
            }
        },
        "features": {
            "gradcam": ENABLE_GRADCAM,
            "liveness_detection": config.ENABLE_LIVENESS,
            "deepfake_check": config.ENABLE_DEEPFAKE_CHECK,
            "early_exit": config.ENABLE_EARLY_EXIT
        },
        "configuration": {
            "frame_interval": config.FRAME_INTERVAL,
            "max_faces": config.MAX_FACES,
            "deepfake_threshold": config.DEEPFAKE_THRESHOLD,
            "verification_threshold": config.VERIFICATION_THRESHOLD
        },
        "device": {
            "type": str(DEVICE),
            "cuda_available": torch.cuda.is_available()
        },
        "timestamp": time.time()
    }

@app.post("/api/v1/detect/deepfake")
async def detect_deepfake(
    file: UploadFile = File(...),
    include_visualization: bool = True
):
    """
    Detect deepfakes in uploaded images
    
    Args:
        file: Image file (JPEG, PNG)
        include_visualization: Whether to include Grad-CAM visualization
    
    Returns:
        JSON with detection results and optional visualizations
    """
    start_time = time.time()
    
    try:
        # Read and validate image
        contents = await file.read()
        try:
            original = Image.open(io.BytesIO(contents)).convert('RGB')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
        # Preprocess image
        input_tensor = tensor_from_pil(original)
        
        # Run inference
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.sigmoid(output).item()
        
        # Interpret results
        is_fake = prob > 0.5
        predicted_class = "Fake" if is_fake else "Real"
        confidence = float(prob if is_fake else (1.0 - prob))
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Base response
        response = {
            "is_authentic": not is_fake,
            "confidence": round(confidence, 4),
            "label": predicted_class,
            "probability_fake": round(prob, 4),
            "probability_real": round(1.0 - prob, 4),
            "model": "EfficientNetB4",
            "processing_time_ms": round(processing_time, 2),
            "timestamp": time.time()
        }
        
        # Add Grad-CAM visualization if requested and enabled
        if include_visualization and ENABLE_GRADCAM and cam is not None:
            try:
                # Generate Grad-CAM
                target_category = 1 if is_fake else 0
                targets = [BinaryClassifierOutputTarget(target_category)]
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
                
                # Prepare visualizations
                original_resized = original.resize((380, 380))
                original_np = np.array(original_resized, dtype=np.float32) / 255.0
                
                # Create overlay
                overlay_rgb = show_cam_on_image(original_np, grayscale_cam, use_rgb=True)
                
                # Create heatmap
                heatmap_uint8 = (grayscale_cam * 255).astype(np.uint8)
                heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
                
                # Encode images as base64
                response["visualizations"] = {
                    "original": to_base64_png(np.array(original_resized)),
                    "gradcam_heatmap": to_base64_png(heatmap_rgb),
                    "overlay": to_base64_png(overlay_rgb)
                }
            except Exception as e:
                response["visualization_error"] = str(e)
        
        return JSONResponse(content=response)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/predict")
async def predict_legacy(file: UploadFile = File(...)):
    """
    Legacy endpoint for backward compatibility - NOW ACCEPTS VIDEO FILES
    
    Processes video frames and returns aggregated deepfake detection results.
    Similar to /verify-with-embedding but without face verification.
    
    Args:
        file: Video file (mp4, avi, mov, mkv)
    
    Returns:
        JSON with detection results and optional Grad-CAM visualizations
    """
    start_time = time.time()
    
    # Validate video processor is available
    if not video_processor:
        return JSONResponse(
            status_code=503, 
            content={"error": "Video processing system not available"}
        )
    
    # Validate file type
    if not file.content_type.startswith('video/') and not file.filename.lower().endswith(
        ('.mp4', '.avi', '.mov', '.mkv')
    ):
        return JSONResponse(
            status_code=400, 
            content={"error": f"Only video files supported. Got: {file.content_type}"}
        )
    
    tmp_path = None
    try:
        # Save video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            content = await file.read()
            if len(content) == 0:
                return JSONResponse(status_code=400, content={"error": "Empty video file"})
            
            tmp.write(content)
            tmp_path = tmp.name
            logger.info(f"📹 Processing video for deepfake detection: {file.filename}")
        
        # Process video to extract frames
        video_result = video_processor.process_video(
            video_path=tmp_path,
            frame_interval=config.FRAME_INTERVAL,
            max_faces=config.MAX_FACES,
            enable_liveness=False  # No liveness check for this endpoint
        )
        
        # Check if frames were extracted
        if video_result['frames_380x380'] is None:
            error_msg = video_result.get('error', "No faces detected in video")
            return JSONResponse(status_code=400, content={"error": error_msg})
        
        frames_380 = video_result['frames_380x380']
        num_frames = frames_380.shape[0]
        logger.info(f"✅ Extracted {num_frames} frames for deepfake analysis")
        
        # Run deepfake detection on all frames
        with torch.no_grad():
            outputs = model(frames_380)
            probabilities = torch.sigmoid(outputs).cpu().numpy().flatten()
        
        # Calculate aggregated results
        avg_prob_fake = float(np.mean(probabilities))
        max_prob_fake = float(np.max(probabilities))
        min_prob_fake = float(np.min(probabilities))
        
        predicted_class = "Fake" if avg_prob_fake > 0.8 else "Real"
        confidence = float(avg_prob_fake if predicted_class == "Fake" else (1.0 - avg_prob_fake))
        
        processing_time = (time.time() - start_time) * 1000
        
        response = {
            "label": predicted_class,
            "confidence": confidence,
            "probability_fake": round(avg_prob_fake, 4),
            "probability_real": round(1.0 - avg_prob_fake, 4),
            "frames_analyzed": num_frames,
            "frame_probabilities": {
                "avg": round(avg_prob_fake, 4),
                "max": round(max_prob_fake, 4),
                "min": round(min_prob_fake, 4)
            },
            "processing_time_ms": round(processing_time, 2)
        }
        
        # Add Grad-CAM for the most suspicious frame if enabled
        if ENABLE_GRADCAM and cam is not None and predicted_class == "Fake":
            try:
                # Find the frame with highest fake probability
                most_suspicious_idx = int(np.argmax(probabilities))
                suspicious_frame = frames_380[most_suspicious_idx:most_suspicious_idx+1]
                
                target_category = 1  # Fake class
                targets = [BinaryClassifierOutputTarget(target_category)]
                grayscale_cam = cam(input_tensor=suspicious_frame, targets=targets)[0]
                
                # Convert frame tensor to numpy for visualization
                frame_np = suspicious_frame[0].cpu().numpy()
                # Denormalize: reverse transforms.Normalize
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                frame_np = frame_np.transpose(1, 2, 0)  # CHW -> HWC
                frame_np = frame_np * std + mean
                frame_np = np.clip(frame_np, 0, 1).astype(np.float32)
                
                overlay_rgb = show_cam_on_image(frame_np, grayscale_cam, use_rgb=True)
                heatmap_uint8 = (grayscale_cam * 255).astype(np.uint8)
                heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
                
                # Convert frame to uint8 for base64 encoding
                frame_uint8 = (frame_np * 255).astype(np.uint8)
                
                response["images"] = {
                    "original": to_base64_png(frame_uint8),
                    "gradcam_heatmap": to_base64_png(heatmap_rgb),
                    "overlay": to_base64_png(overlay_rgb)
                }
                response["gradcam_frame_index"] = most_suspicious_idx
                response["gradcam_frame_probability"] = round(max_prob_fake, 4)
            except Exception as e:
                logger.warning(f"⚠️ Grad-CAM generation failed: {e}")
        
        logger.info(f"✅ Video deepfake detection complete: {predicted_class} ({confidence:.4f})")
        return response
    
    except Exception as e:
        logger.error(f"❌ Predict error: {str(e)}")
        return JSONResponse(
            status_code=500, 
            content={"error": f"Processing error: {str(e)}"}
        )
    
    finally:
        # Cleanup temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

@app.post("/api/v1/verify/liveness")
async def verify_liveness(file: UploadFile = File(...)):
    """
    Liveness detection endpoint (placeholder for future implementation)
    Currently performs deepfake detection only
    """
    result = await detect_deepfake(file, include_visualization=False)
    
    # Extract data from JSONResponse
    if isinstance(result, JSONResponse):
        import json
        data = json.loads(result.body.decode())
    else:
        data = result
    
    return {
        "liveness_passed": data.get("is_authentic", False),
        "deepfake_passed": data.get("is_authentic", False),
        "confidence": data.get("confidence", 0.0),
        "note": "Full liveness detection coming soon. Currently using deepfake detection."
    }

@app.post("/extract")
async def extract_features(file: UploadFile = File(...)):
    """
    Extract 512-d FaceNet embeddings from a video and return
    compressed float32 Base64 for efficient storage.
    """
    if not video_processor or not facenet_model:
        raise HTTPException(
            status_code=503,
            detail="Feature extraction system not available"
        )

    # Validate file type
    if not file.content_type.startswith("video/") and not file.filename.lower().endswith(
        (".mp4", ".avi", ".mov", ".mkv")
    ):
        raise HTTPException(status_code=400, detail="Only video files supported")

    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        tmp.write(content)
        tmp_path = tmp.name

    try:
        logger.info(f"📹 Extracting features from: {file.filename}")

        # Run video processing
        video_result = video_processor.process_video(
            video_path=tmp_path,
            frame_interval=config.FRAME_INTERVAL,
            max_faces=config.MAX_FACES,
            enable_liveness=config.ENABLE_LIVENESS
        )

        # Liveness check
        if config.ENABLE_LIVENESS and not video_result["liveness_passed"]:
            raise HTTPException(
                status_code=403,
                detail=f"Liveness check failed. Detected {video_result['total_blinks']} blink(s)"
            )

        # No face detected
        if video_result["faces_160x160"] is None:
            raise HTTPException(
                status_code=400,
                detail=video_result.get("error", "No faces detected")
            )

        # Extract FaceNet embeddings
        with torch.no_grad():
            embeddings = facenet_model(video_result["faces_160x160"])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            avg_embedding = embeddings.mean(dim=0).cpu().numpy()  # float64 by default

        logger.info(f"✅ Extracted embedding from {video_result['num_faces']} faces")

        # Convert to float32 for smaller storage
        embedding_float32 = avg_embedding.astype(np.float32)

        # Convert to raw bytes
        bytes_data = embedding_float32.tobytes()

        # Base64 encode
        embedding_b64 = base64.b64encode(bytes_data).decode("utf-8")

        # Return lightweight response
        return {
            "success": True,
            "embedding_b64": embedding_b64,
            "dtype": "float32",
            "shape": [512],
            "byte_length": len(bytes_data),
            "faces_processed": video_result["num_faces"],
            "liveness_check": {
                "passed": video_result["liveness_passed"],
                "blinks_detected": video_result["total_blinks"]
            } if config.ENABLE_LIVENESS else None,
            "message": "Embedding extracted successfully. Use embedding_b64 for storage."
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Extract features error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

    finally:
        # Cleanup temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/verify-two-videos")
async def verify_two_videos(
    file1: UploadFile = File(..., description="First video file"),
    file2: UploadFile = File(..., description="Second video file"),
    threshold: float = Form(0.6, description="Similarity threshold (0.0-1.0)")
):
    """
    Verify if two videos contain the same person using FaceNet
    
    Compares facial embeddings from two videos to determine if they show the same person.
    
    Args:
        file1: First video file
        file2: Second video file
        threshold: Similarity threshold (default: 0.6)
    
    Returns:
        Similarity score and verification result
    """
    if not video_processor or not facenet_model:
        raise HTTPException(
            status_code=503,
            detail="Video verification system not available"
        )
    
    try:
        # Validate files
        for file in [file1, file2]:
            if not file.content_type.startswith('video/') and not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                raise HTTPException(status_code=400, detail="Only video files supported")
        
        if not 0.0 <= threshold <= 1.0:
            raise HTTPException(status_code=400, detail="Threshold must be between 0.0 and 1.0")
        
        temp_files = []
        try:
            # Save both files
            for file in [file1, file2]:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    content = await file.read()
                    if len(content) == 0:
                        raise HTTPException(status_code=400, detail=f"File {file.filename} is empty")
                    tmp.write(content)
                    temp_files.append(tmp.name)
            
            logger.info(f"📹 Comparing videos: {file1.filename} vs {file2.filename}")
            
            # Process both videos
            embeddings = []
            for i, tmp_path in enumerate(temp_files):
                video_result = video_processor.process_video(
                    video_path=tmp_path,
                    frame_interval=config.FRAME_INTERVAL,
                    max_faces=config.MAX_FACES,
                    enable_liveness=config.ENABLE_LIVENESS
                )
                
                if config.ENABLE_LIVENESS and not video_result['liveness_passed']:
                    raise HTTPException(
                        status_code=403,
                        detail=f"Liveness check failed for video {i+1}"
                    )
                
                if video_result['faces_160x160'] is None:
                    raise HTTPException(
                        status_code=400,
                        detail=f"No faces detected in video {i+1}"
                    )
                
                # Extract embedding
                with torch.no_grad():
                    emb = facenet_model(video_result['faces_160x160'])
                    emb = F.normalize(emb, p=2, dim=1)
                    avg_emb = emb.mean(dim=0).cpu().numpy()
                    embeddings.append(avg_emb)
            
            # Calculate similarity
            similarity = float(np.dot(embeddings[0], embeddings[1]))
            similarity = np.clip(similarity, -1.0, 1.0)
            is_match = similarity >= threshold
            
            result = "same_person" if is_match else "different_person"
            
            logger.info(f"✅ Comparison complete. Similarity: {similarity:.4f}, Result: {result}")
            
            return {
                "success": True,
                "similarity": round(similarity, 4),
                "threshold_used": threshold,
                "result": result,
                "is_match": is_match,
                "model": "FaceNet (InceptionResnetV1)",
                "message": "Verification completed successfully"
            }
        
        finally:
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Verify two videos error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


def is_normalized(embedding: np.ndarray, tolerance: float = 0.01) -> bool:
    """
    Check if an embedding vector is already normalized (L2 norm ≈ 1.0)
    
    Args:
        embedding: Numpy array to check
        tolerance: Acceptable deviation from 1.0 (default: 0.01)
    
    Returns:
        True if embedding is normalized, False otherwise
    """
    norm = np.linalg.norm(embedding)
    return abs(norm - 1.0) < tolerance


@app.post("/verify-with-embedding")
async def verify_with_embedding(
    file: UploadFile = File(..., description="Video file to verify"),
    stored_embedding: str = Form(..., description="Stored 512-dim FaceNet embedding as JSON string"),
    threshold: float = Form(0.6, description="Face verification threshold (0.0-1.0)")
):
    """
    **UNIFIED VERIFICATION ENDPOINT**
    
    Performs sequential verification with early exit optimization:
    1. **Deepfake Detection** (EfficientNetB4) - Checks if video is AI-generated
    2. **Liveness Detection** (Optional) - Verifies human presence via blink detection
    3. **Face Verification** (FaceNet) - Matches against stored embedding
    
    **Early Exit**: If deepfake detected, returns immediately without face verification.
    
    Args:
        file: Video file (mp4, avi, mov, mkv)
        stored_embedding: Pre-stored 512-dimensional FaceNet embedding (JSON array)
        threshold: Similarity threshold for face matching (default: 0.6)
    
    Returns:
        Complete verification result with deepfake check, liveness check, and face verification
    
    Example Response (Success):
    {
        "success": true,
        "result": "verified",
        "message": "All checks passed. User authenticated successfully.",
        "deepfake_check": {
            "is_authentic": true,
            "probability_fake": 0.1234,
            "probability_real": 0.8766,
            "confidence": 0.8766,
            "num_faces_analyzed": 25
        },
        "liveness_check": {
            "passed": true,
            "blinks_detected": 3
        },
        "face_verification": {
            "is_match": true,
            "similarity": 0.8750,
            "threshold_used": 0.6
        },
        "processing_time_ms": 15234.56,
        "performance_breakdown": {
            "video_processing_ms": 2842.76,
            "deepfake_detection_ms": 8961.47,
            "face_verification_ms": 430.33
        },
        "model_info": {
            "deepfake_model": "EfficientNetB4",
            "verification_model": "FaceNet (InceptionResnetV1)",
            "file_processed": "video.mp4",
            "faces_analyzed": 25
        }
    }
    
    Example Response (Deepfake Detected):
    {
        "success": false,
        "result": "deepfake_detected",
        "message": "Video contains AI-generated or manipulated content. Verification rejected.",
        "deepfake_check": {
            "is_authentic": false,
            "probability_fake": 0.9234,
            "probability_real": 0.0766,
            "confidence": 0.9234,
            "num_faces_analyzed": 25
        },
        "liveness_check": null,
        "face_verification": null,
        "processing_time_ms": 9234.56,
        "performance_breakdown": {
            "video_processing_ms": 2842.76,
            "deepfake_detection_ms": 6391.80
        }
    }
    """
    start_time = time.time()
    
    # VALIDATION
    
    if not video_processor or not facenet_model:
        logger.error("❌ Required models not loaded")
        raise HTTPException(
            status_code=503,
            detail="Video verification system not available. Required models not loaded."
        )
    
    try:
        # Validate file type
        if not file.content_type.startswith('video/') and not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            logger.warning(f"⚠️ Invalid file type: {file.content_type}")
            raise HTTPException(
                status_code=400,
                detail=f"Only video files supported. Got: {file.content_type}"
            )
        
        # Parse stored embedding
        try:
            emb_bytes = base64.b64decode(stored_embedding)
        
            if len(emb_bytes) != 2048:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid embedding size. Expected 2048 bytes, got {len(emb_bytes)}"
                )
        
            stored_emb = np.frombuffer(emb_bytes, dtype=np.float32)
        
            if stored_emb.shape[0] != 512:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid embedding dimension. Expected 512 floats, got {stored_emb.shape[0]}"
                )
        
            logger.info(f"✅ Loaded stored embedding from Base64, shape={stored_emb.shape}")
        
        except Exception as e:
            logger.error(f"❌ Error decoding embedding: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid Base64 embedding: {e}")
        
        # Validate embedding dimensions
        if len(stored_emb) != 512:
            logger.error(f"❌ Invalid embedding dimension: {len(stored_emb)}")
            raise HTTPException(
                status_code=400,
                detail=f"Embedding must be 512-dimensional, got: {len(stored_emb)}"
            )
        
        # Validate threshold
        if not 0.0 <= threshold <= 1.0:
            logger.error(f"❌ Invalid threshold: {threshold}")
            raise HTTPException(
                status_code=400,
                detail="Threshold must be between 0.0 and 1.0"
            )
        
        logger.info(f"✅ All validations passed")
        logger.info(f"   - File: {file.filename}")
        logger.info(f"   - Embedding: 512-dimensional")
        logger.info(f"   - Threshold: {threshold}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    # SAVE VIDEO TEMPORARILY
    
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            content = await file.read()
            if len(content) == 0:
                logger.error("❌ Empty video file")
                raise HTTPException(status_code=400, detail="Empty video file")
            
            tmp.write(content)
            tmp_path = tmp.name
            logger.info(f"✅ Video saved temporarily: {tmp_path} ({len(content)} bytes)")
        
        # STEP 1: VIDEO PROCESSING (Extract frames & faces)
        
        logger.info(f"🎬 Processing video: {file.filename}")
        processing_start = time.time()
        
        video_result = video_processor.process_video(
            video_path=tmp_path,
            frame_interval=config.FRAME_INTERVAL,
            max_faces=config.MAX_FACES,
            enable_liveness=config.ENABLE_LIVENESS
        )
        
        processing_time = (time.time() - processing_start) * 1000
        logger.info(f"✅ Video processed in {processing_time:.2f}ms")
        logger.info(f"   - Video result keys: {list(video_result.keys())}")
        logger.info(f"   - Faces detected: {video_result.get('num_faces', 'N/A')}")
        
        # STEP 2: LIVENESS CHECK (Optional)
        
        if config.ENABLE_LIVENESS:
            logger.info(f"👁️  Checking liveness (blinks required: {config.MIN_BLINKS_FOR_LIVENESS})...")
            if not video_result['liveness_passed']:
                total_time = (time.time() - start_time) * 1000
                logger.warning(
                    f"❌ LIVENESS FAILED! Blinks detected: {video_result['total_blinks']} "
                    f"(required: {config.MIN_BLINKS_FOR_LIVENESS})"
                )
                
                return {
                    "success": False,
                    "result": "liveness_failed",
                    "message": f"Liveness check failed. Detected {video_result['total_blinks']} blink(s), "
                              f"required {config.MIN_BLINKS_FOR_LIVENESS}.",
                    "deepfake_check": None,
                    "liveness_check": {
                        "passed": False,
                        "blinks_detected": int(video_result['total_blinks']),
                        "blinks_required": config.MIN_BLINKS_FOR_LIVENESS
                    },
                    "face_verification": None,
                    "processing_time_ms": round(total_time, 2),
                    "performance_breakdown": {
                        "video_processing_ms": round(processing_time, 2)
                    }
                }
            
            logger.info(f"✅ Liveness check passed ({video_result['total_blinks']} blinks detected)")
        
        # STEP 3: CHECK FOR FACES
        
        if video_result['faces_160x160'] is None or video_result['frames_380x380'] is None:
            total_time = (time.time() - start_time) * 1000
            error_msg = video_result.get('error', "No faces detected in video")
            logger.error(f"❌ {error_msg}")
            
            raise HTTPException(
                status_code=400,
                detail=error_msg
            )
        
        faces_160 = video_result['faces_160x160']
        frames_380 = video_result['frames_380x380']
        
        logger.info(f"✅ Face data retrieved")
        logger.info(f"   - 160x160 faces: {faces_160.shape}")
        logger.info(f"   - 380x380 frames: {frames_380.shape}")
        
        # STEP 4: DEEPFAKE DETECTION (Early Exit if Fake)
        
        logger.info(f"🔍 Running deepfake detection (Full Frame Analysis)...")
        deepfake_start = time.time()
        
        with torch.no_grad():
            outputs = model(frames_380)
            probabilities = torch.sigmoid(outputs).cpu().numpy().flatten()
        
        avg_prob_fake = float(np.mean(probabilities))
        is_authentic = avg_prob_fake < config.DEEPFAKE_THRESHOLD
        
        deepfake_time = (time.time() - deepfake_start) * 1000
        
        deepfake_result = {
            'is_authentic': bool(is_authentic),
            'probability_fake': round(float(avg_prob_fake), 4),
            'probability_real': round(float(1.0 - avg_prob_fake), 4),
            'confidence': round(
                float((1.0 - avg_prob_fake) if is_authentic else avg_prob_fake),
                4
            ),
            'num_faces_analyzed': int(len(probabilities))
        }
        
        logger.info(f"✅ Deepfake detection complete")
        logger.info(f"   - Is Authentic: {is_authentic}")
        logger.info(f"   - Probability (Fake): {avg_prob_fake:.4f}")
        logger.info(f"   - Probability (Real): {1.0 - avg_prob_fake:.4f}")
        logger.info(f"   - Threshold: {config.DEEPFAKE_THRESHOLD}")
        logger.info(f"   - Frames analyzed: {len(probabilities)}")
        
        # EARLY EXIT: If deepfake detected
        
        if not is_authentic:
            total_time = (time.time() - start_time) * 1000
            
            logger.warning(f"⚠️ DEEPFAKE DETECTED! Probability: {avg_prob_fake:.4f}")
            logger.warning(f"   Stopping verification pipeline - returning early exit")
            
            return {
                "success": False,
                "result": "deepfake_detected",
                "message": "Video contains AI-generated or manipulated content. Verification rejected.",
                "deepfake_check": deepfake_result,
                "liveness_check": {
                    "passed": bool(video_result['liveness_passed']),
                    "blinks_detected": int(video_result['total_blinks'])
                } if config.ENABLE_LIVENESS else None,
                "face_verification": None,
                "processing_time_ms": round(total_time, 2),
                "performance_breakdown": {
                    "video_processing_ms": round(processing_time, 2),
                    "deepfake_detection_ms": round(deepfake_time, 2)
                }
            }
        
        logger.info(f"✅ Deepfake check passed!")
        
        # STEP 5: FACE VERIFICATION (Only if deepfake check passed)
        
        logger.info(f"🔐 Running face verification...")
        verification_start = time.time()
        
        # Extract embeddings
        with torch.no_grad():
            embeddings = facenet_model(faces_160)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            avg_embedding = embeddings.mean(dim=0).cpu().numpy()
        
        logger.info(f"✅ Embeddings extracted and normalized")
        logger.info(f"   - Number of face embeddings: {len(embeddings)}")
        
        # EMBEDDING NORMALIZATION (CRITICAL!)
        
        # SMART NORMALIZATION: Prevent double normalization issues
        # Check if stored embedding is already normalized using helper function
        stored_emb_norm_value = np.linalg.norm(stored_emb)
        logger.info(f"📊 Stored embedding normalization check")
        logger.info(f"   - Original norm: {stored_emb_norm_value:.6f}")
        
        if is_normalized(stored_emb, tolerance=0.01):
            logger.info(f"✅ Stored embedding already normalized (norm={stored_emb_norm_value:.6f}), using as-is")
            stored_emb_normalized = stored_emb
        else:
            logger.info(f"🔧 Stored embedding NOT normalized (norm={stored_emb_norm_value:.6f}), applying normalization")
            stored_emb_normalized = stored_emb / stored_emb_norm_value
        
        # Extracted embedding: mean of normalized vectors may not be normalized
        # Always check and normalize if needed
        avg_embedding_norm = np.linalg.norm(avg_embedding)
        logger.info(f"📊 Extracted embedding normalization check")
        logger.info(f"   - Norm after averaging: {avg_embedding_norm:.6f}")
        
        if avg_embedding_norm > 0:
            if is_normalized(avg_embedding, tolerance=0.01):
                logger.info(f"✅ Extracted embedding already normalized, using as-is")
                avg_embedding_normalized = avg_embedding
            else:
                logger.info(f"🔧 Extracted embedding NOT normalized, applying normalization")
                avg_embedding_normalized = avg_embedding / avg_embedding_norm
        else:
            logger.error(f"❌ Extracted embedding has zero norm!")
            raise Exception("Invalid embedding norm")
        
        logger.info(f"✅ Final normalization status")
        logger.info(f"   - Stored embedding norm: {np.linalg.norm(stored_emb_normalized):.6f}")
        logger.info(f"   - Extracted embedding norm: {np.linalg.norm(avg_embedding_normalized):.6f}")
        
        # CALCULATE SIMILARITY
        
        similarity = float(np.dot(stored_emb_normalized, avg_embedding_normalized))
        similarity = np.clip(similarity, -1.0, 1.0)
        is_match = similarity >= threshold
        
        verification_time = (time.time() - verification_start) * 1000
        
        logger.info(f"✅ Face verification complete")
        logger.info(f"   - Similarity: {similarity:.4f}")
        logger.info(f"   - Threshold: {threshold}")
        logger.info(f"   - Is Match: {is_match}")
        
        # COMPILE FINAL RESULT
        
        total_time = (time.time() - start_time) * 1000
        
        if is_match:
            logger.info(f"✅ VERIFICATION SUCCESSFUL!")
            result_status = "verified"
            result_message = "All checks passed. User authenticated successfully."
            success = True
        else:
            logger.warning(f"❌ FACE MISMATCH! Similarity: {similarity:.4f}")
            result_status = "face_mismatch"
            result_message = "Face does not match stored embedding. Authentication failed."
            success = False
        
        response_data = {
            "success": bool(success),
            "result": result_status,
            "message": result_message,
            "deepfake_check": deepfake_result,
            "liveness_check": {
                "passed": bool(video_result['liveness_passed']),
                "blinks_detected": int(video_result['total_blinks'])
            } if config.ENABLE_LIVENESS else None,
            "face_verification": {
                "is_match": bool(is_match),
                "similarity": round(float(similarity), 4),
                "threshold_used": float(threshold)
            },
            "processing_time_ms": round(float(total_time), 2),
            "performance_breakdown": {
                "video_processing_ms": round(float(processing_time), 2),
                "deepfake_detection_ms": round(float(deepfake_time), 2),
                "face_verification_ms": round(float(verification_time), 2)
            },
            "model_info": {
                "deepfake_model": "EfficientNetB4",
                "verification_model": "FaceNet (InceptionResnetV1)",
                "file_processed": file.filename,
                "faces_analyzed": int(video_result['num_faces'])
            }
        }
        
        logger.info(f"📊 Final response compiled:")
        logger.info(f"   - Success: {success}")
        logger.info(f"   - Result: {result_status}")
        logger.info(f"   - Total time: {total_time:.2f}ms")
        
        return response_data
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Verification error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
    
    finally:
        # CLEANUP
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                logger.info(f"🧹 Temporary file cleaned up: {tmp_path}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to cleanup temp file: {e}")

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("="*60)
    logger.info("🚀 Unified Deepfake Detection & Face Verification API Started")
    logger.info(f"📦 Deepfake Model: EfficientNetB4")
    logger.info(f"🔐 Face Verification Model: {'FaceNet (InceptionResnetV1)' if facenet_model else 'Disabled'}")
    logger.info(f"🖥️  Device: {DEVICE}")
    logger.info(f"🎨 Grad-CAM: {'Enabled' if ENABLE_GRADCAM else 'Disabled'}")
    logger.info(f"👁️  Liveness Detection: {'Enabled' if config.ENABLE_LIVENESS else 'Disabled'}")
    logger.info("="*60)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)