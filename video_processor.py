"""
Optimized Video Processor for Unified Deepfake Detection and Face Verification
Extracts faces at multiple resolutions simultaneously for efficiency
"""
import cv2
import torch
import numpy as np
from PIL import Image
import logging
from typing import Tuple, Optional, Dict, Any
from facenet_pytorch import MTCNN
from torchvision import transforms
import mediapipe as mp
from scipy.spatial import distance as dist
import config

logger = logging.getLogger(__name__)

class OptimizedVideoProcessor:
    """
    Optimized video processor that extracts faces at multiple resolutions.
    
    Features:
    - Single-pass video processing
    - Dual-resolution face extraction (160x160 and 380x380)
    - Liveness detection (blink detection)
    - Efficient frame sampling
    """
    
    def __init__(self, device: torch.device):
        """Initialize video processor with MTCNN and MediaPipe"""
        self.device = device
        self.mtcnn = MTCNN(
            image_size=160, 
            margin=0, 
            min_face_size=20, 
            device=device
        )
        
        # MediaPipe for liveness detection
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # Eye landmark indices for blink detection
        self.LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        
        # Preprocessing for EfficientNetB4
        self.efficientnet_transform = transforms.Compose([
            transforms.Resize((config.EFFICIENTNET_IMAGE_SIZE, config.EFFICIENTNET_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        logger.info("✅ Optimized video processor initialized")
    
    def _calculate_ear(self, eye_points: np.ndarray) -> float:
        """Calculate Eye Aspect Ratio for blink detection"""
        A = dist.euclidean(eye_points[1], eye_points[5])
        B = dist.euclidean(eye_points[2], eye_points[4])
        C = dist.euclidean(eye_points[0], eye_points[3])
        ear = (A + B) / (2.0 * C)
        return ear
    
    def process_video(
        self,
        video_path: str,
        frame_interval: int = None,
        max_faces: int = None,
        enable_liveness: bool = None
    ) -> Dict[str, Any]:
        """
        Process video and extract faces at multiple resolutions.
        
        Returns dict with:
        - faces_160x160: Tensor for FaceNet (N, 3, 160, 160)
        - faces_380x380: Tensor for EfficientNetB4 (N, 3, 380, 380)
        - liveness_passed: Boolean
        - total_blinks: Number of blinks
        - num_faces: Number of faces extracted
        """
        frame_interval = frame_interval or config.FRAME_INTERVAL
        max_faces = max_faces or config.MAX_FACES
        enable_liveness = enable_liveness if enable_liveness is not None else config.ENABLE_LIVENESS
        
        # Liveness state
        eye_closed_counter = 0
        total_blinks = 0
        liveness_passed = False
        
        # Face storage
        faces_160 = []
        faces_380 = []
        frames_380 = []  # Store full frames for deepfake detection
        
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        faces_detected = 0
        
        logger.info(f"Processing video - Interval: {frame_interval}, Max faces: {max_faces}, Liveness: {enable_liveness}")
        
        # Initialize MediaPipe (CPU mode for Docker compatibility)
        if enable_liveness:
            face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                static_image_mode=False
            )
            # Note: MediaPipe will automatically use CPU in Docker environments
            # where GPU/OpenGL context is not available
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % frame_interval == 0:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Liveness detection
                    if enable_liveness and not liveness_passed:
                        results = face_mesh.process(rgb_frame)
                        
                        if results.multi_face_landmarks:
                            landmarks = results.multi_face_landmarks[0]
                            h, w, c = frame.shape
                            
                            points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks.landmark]
                            
                            left_eye_pts = np.array([points[i] for i in self.LEFT_EYE_INDICES])
                            right_eye_pts = np.array([points[i] for i in self.RIGHT_EYE_INDICES])
                            
                            left_ear = self._calculate_ear(left_eye_pts)
                            right_ear = self._calculate_ear(right_eye_pts)
                            ear = (left_ear + right_ear) / 2.0
                            
                            if ear < config.EAR_THRESHOLD:
                                eye_closed_counter += 1
                            else:
                                if eye_closed_counter >= config.CONSECUTIVE_FRAMES:
                                    total_blinks += 1
                                    logger.info(f"✅ Blink detected! Total: {total_blinks}")
                                eye_closed_counter = 0
                            
                            if total_blinks >= config.MIN_BLINKS_FOR_LIVENESS:
                                liveness_passed = True
                    
                    # Extract faces once liveness confirmed (or if disabled)
                    if not enable_liveness or liveness_passed:
                        pil_image = Image.fromarray(rgb_frame)
                        
                        try:
                            # MTCNN extracts 160x160 face
                            face_160 = self.mtcnn(pil_image)
                            
                            if face_160 is not None:
                                if face_160.dim() == 3:
                                    face_160 = face_160.unsqueeze(0)
                                
                                faces_160.append(face_160)
                                
                                # Create 380x380 version (Cropped Face)
                                face_160_np = face_160.squeeze(0).permute(1, 2, 0).cpu().numpy()
                                face_160_np = ((face_160_np + 1) * 127.5).astype(np.uint8)
                                face_pil = Image.fromarray(face_160_np)
                                
                                face_380 = self.efficientnet_transform(face_pil).unsqueeze(0)
                                faces_380.append(face_380)
                                
                                # Create 380x380 version (Full Frame)
                                frame_380 = self.efficientnet_transform(pil_image).unsqueeze(0)
                                frames_380.append(frame_380)
                                
                                faces_detected += 1
                                
                        except Exception as e:
                            logger.warning(f"Could not extract face in frame {frame_idx}: {str(e)}")
                
                frame_idx += 1
                if faces_detected >= max_faces:
                    break
            
        finally:
            cap.release()
            if enable_liveness:
                face_mesh.close()
        
        logger.info(f"Processing complete. Faces: {faces_detected}, Blinks: {total_blinks}")
        
        # Check results
        if enable_liveness and not liveness_passed:
            return {
                'faces_160x160': None,
                'faces_380x380': None,
                'frames_380x380': None,
                'liveness_passed': False,
                'total_blinks': total_blinks,
                'num_faces': 0,
                'error': 'Liveness check failed'
            }
        
        if len(faces_160) == 0:
            return {
                'faces_160x160': None,
                'faces_380x380': None,
                'frames_380x380': None,
                'liveness_passed': liveness_passed if enable_liveness else True,
                'total_blinks': total_blinks,
                'num_faces': 0,
                'error': 'No faces detected'
            }
        
        # Concatenate batches
        faces_160_batch = torch.cat(faces_160, dim=0).to(self.device)
        faces_380_batch = torch.cat(faces_380, dim=0).to(self.device)
        frames_380_batch = torch.cat(frames_380, dim=0).to(self.device)
        
        return {
            'faces_160x160': faces_160_batch,
            'faces_380x380': faces_380_batch,
            'frames_380x380': frames_380_batch,
            'liveness_passed': liveness_passed if enable_liveness else True,
            'total_blinks': total_blinks,
            'num_faces': faces_detected
        }
