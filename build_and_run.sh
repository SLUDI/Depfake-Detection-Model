#!/bin/bash
# Build and Run Script for Unified Deepfake Detection & Face Verification API

echo "🔨 Building Docker image..."
docker build -t deepfake-detector:latest .

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo ""
    echo "🚀 Starting container..."
    
    # Stop and remove existing container if running
    docker stop deepfake-detector 2>/dev/null
    docker rm deepfake-detector 2>/dev/null
    
    # Run the container
    docker run -d \
        --name deepfake-detector \
        --gpus all \
        -p 8080:8080 \
        -v $(pwd)/model:/app/model \
        -e DEEPFAKE_THRESHOLD=0.5 \
        -e VERIFICATION_THRESHOLD=0.6 \
        -e FRAME_INTERVAL=5 \
        -e MAX_FACES=30 \
        -e ENABLE_LIVENESS=false \
        -e ENABLE_DEEPFAKE_CHECK=true \
        -e ENABLE_GRADCAM=false \
        deepfake-detector:latest
    
    echo "✅ Container started!"
    echo ""
    echo "📊 Checking logs..."
    sleep 3
    docker logs deepfake-detector
    
    echo ""
    echo "🌐 API is available at: http://localhost:8080"
    echo "📚 API Documentation: http://localhost:8080/docs"
    echo ""
    echo "To view logs: docker logs -f deepfake-detector"
    echo "To stop: docker stop deepfake-detector"
else
    echo "❌ Build failed!"
    exit 1
fi
