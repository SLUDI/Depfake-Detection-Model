
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

## ☁️ Cloud Deployment

### Google Cloud Run

```bash
# Build and deploy
gcloud run deploy deepfake-detector \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 60s \
  --min-instances 1
```

### AWS Lambda (with Container)

```bash
# Build for ARM64 (Graviton)
docker build --platform linux/arm64 -t deepfake-detector .

# Tag and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker tag deepfake-detector:latest <account>.dkr.ecr.us-east-1.amazonaws.com/deepfake-detector:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/deepfake-detector:latest
```

### Render.com

1. Connect your GitHub repository
2. Select "Docker" as environment
3. Set port to `8080`
4. Deploy

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_PATH` | Path to model file | `/app/model/effb4_ai_detector.pt` |
| `ENABLE_GRADCAM` | Enable Grad-CAM visualization | `true` |
| `HF_MODEL_REPO` | HuggingFace repo for model download | None |
| `HF_TOKEN` | HuggingFace API token | None |

### Disable Grad-CAM for Faster Inference

```bash
export ENABLE_GRADCAM=false
uvicorn app:app --host 0.0.0.0 --port 8080
```

## 📱 Mobile App Integration

### React Native Example

```javascript
const detectDeepfake = async (imageUri) => {
  const formData = new FormData();
  formData.append('file', {
    uri: imageUri,
    type: 'image/jpeg',
    name: 'photo.jpg',
  });

  const response = await fetch('https://your-api.com/api/v1/detect/deepfake', {
    method: 'POST',
    body: formData,
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  const result = await response.json();
  console.log('Is Authentic:', result.is_authentic);
  console.log('Confidence:', result.confidence);
};
```

### Flutter Example

```dart
Future<Map<String, dynamic>> detectDeepfake(File imageFile) async {
  var request = http.MultipartRequest(
    'POST',
    Uri.parse('https://your-api.com/api/v1/detect/deepfake'),
  );
  
  request.files.add(await http.MultipartFile.fromPath('file', imageFile.path));
  
  var response = await request.send();
  var responseData = await response.stream.bytesToString();
  
  return json.decode(responseData);
}
```

## 🔐 Integration with Digital Identity Wallet

### Hyperledger Fabric Integration

```python
# Example: Create DID with biometric verification
async def create_did_with_verification(user_image):
    # Step 1: Verify image authenticity
    verification = await detect_deepfake(user_image)
    
    if not verification['is_authentic'] or verification['confidence'] < 0.85:
        raise Exception("Biometric verification failed")
    
    # Step 2: Create DID on Hyperledger Fabric
    did = await fabric_client.create_did(
        biometric_hash=hash(user_image),
        verification_proof={
            'method': 'deepfake-detection',
            'model': 'EfficientNetB4',
            'confidence': verification['confidence'],
            'timestamp': time.time()
        }
    )
    
    return did
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
