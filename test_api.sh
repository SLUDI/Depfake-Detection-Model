#!/bin/bash
# test_api.sh - Test script for deepfake detection API

API_URL="${API_URL:-http://localhost:8080}"

echo "🧪 Testing Deepfake Detection API"
echo "=================================="
echo ""

# Test 1: Health Check
echo "1️⃣  Testing health endpoint..."
curl -s "${API_URL}/health" | python3 -m json.tool
echo ""

# Test 2: Root endpoint
echo "2️⃣  Testing root endpoint..."
curl -s "${API_URL}/" | python3 -m json.tool
echo ""

# Test 3: Deepfake detection (requires test image)
if [ -f "test_image.jpg" ]; then
    echo "3️⃣  Testing deepfake detection..."
    curl -s -X POST "${API_URL}/api/v1/detect/deepfake" \
        -F "file=@test_image.jpg" \
        -F "include_visualization=false" | python3 -m json.tool
    echo ""
else
    echo "⚠️  Skipping detection test - test_image.jpg not found"
    echo "   Create a test_image.jpg to test detection"
    echo ""
fi

echo "✅ API tests complete!"
