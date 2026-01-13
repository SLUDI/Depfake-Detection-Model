
# Deepfake Detection API

Production-ready deepfake detection service using **EfficientNetB4** for Digital Identity Wallet integration.

## 🚀 Features

- **EfficientNetB4 Model**: State-of-the-art deepfake detection
- **Fast Inference**: Optimized for production use
- **Grad-CAM Visualization**: Visual explanations of predictions
- **RESTful API**: Easy integration with mobile/web apps
- **Docker Support**: Containerized deployment
- **Health Checks**: Built-in monitoring endpoints

## 📋 API Endpoints

### 1. Health Check
```bash
GET /health
```

### 2. Deepfake Detection (Recommended)
```bash
POST /api/v1/detect/deepfake
Content-Type: multipart/form-data

Parameters:
  - file: image file (JPEG, PNG)
  - include_visualization: boolean (default: true)

Response:
{
  "is_authentic": true,
  "confidence": 0.9234,
  "label": "Real",
  "probability_fake": 0.0766,
  "probability_real": 0.9234,
  "model": "EfficientNetB4",
  "processing_time_ms": 245.67,
  "visualizations": {
    "original": "base64_encoded_image",
    "gradcam_heatmap": "base64_encoded_image",
    "overlay": "base64_encoded_image"
  }
}
```

### 3. Legacy Endpoint (Backward Compatible)
```bash
POST /predict
Content-Type: multipart/form-data

Parameters:
  - file: image file
```

### 4. Liveness Detection
```bash
POST /api/v1/verify/liveness
Content-Type: multipart/form-data

Parameters:
  - file: image file
```

## 🛠️ Installation

### Local Development

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Place your model**:
```bash
# Ensure your model is at: model/effb4_ai_detector.pt
```

3. **Run the server**:
```bash
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

4. **Test the API**:
```bash
curl -X POST "http://localhost:8080/api/v1/detect/deepfake" \
  -F "file=@test_image.jpg"
```

### Docker Deployment

1. **Build the image**:
```bash
docker build -t deepfake-detector .
```

2. **Run the container**:
```bash
docker run -p 8080:8080 deepfake-detector
```

3. **Test**:
```bash
curl http://localhost:8080/health
```

## 🧪 Testing

```bash
# Test with sample image
curl -X POST "http://localhost:8080/api/v1/detect/deepfake" \
  -F "file=@sample_real.jpg" \
  -F "include_visualization=true"

# Health check
curl http://localhost:8080/health
```

## 📝 Model Information

- **Architecture**: EfficientNetB4
- **Input Size**: 380x380 pixels
- **Parameters**: ~19M
- **Model Size**: ~70MB
- **Training**: Fine-tuned on deepfake dataset

---

**Built for Digital Identity Wallet Integration** 🔐
