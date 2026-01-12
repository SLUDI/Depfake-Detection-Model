"""
Example Python client for Deepfake Detection API
Usage: python client_example.py <image_path>
"""

import sys
import requests
import json
from pathlib import Path


class DeepfakeDetectionClient:
    """Client for interacting with Deepfake Detection API"""
    
    def __init__(self, api_url: str = "http://localhost:8080"):
        self.api_url = api_url.rstrip('/')
    
    def health_check(self) -> dict:
        """Check API health status"""
        response = requests.get(f"{self.api_url}/health")
        response.raise_for_status()
        return response.json()
    
    def detect_deepfake(self, image_path: str, include_visualization: bool = False) -> dict:
        """
        Detect deepfakes in an image
        
        Args:
            image_path: Path to image file
            include_visualization: Whether to include Grad-CAM visualization
        
        Returns:
            Detection results as dictionary
        """
        with open(image_path, 'rb') as f:
            files = {'file': f}
            data = {'include_visualization': str(include_visualization).lower()}
            
            response = requests.post(
                f"{self.api_url}/api/v1/detect/deepfake",
                files=files,
                data=data
            )
            response.raise_for_status()
            return response.json()
    
    def verify_liveness(self, image_path: str) -> dict:
        """
        Verify liveness (currently uses deepfake detection)
        
        Args:
            image_path: Path to image file
        
        Returns:
            Liveness verification results
        """
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f"{self.api_url}/api/v1/verify/liveness",
                files=files
            )
            response.raise_for_status()
            return response.json()


def main():
    if len(sys.argv) < 2:
        print("Usage: python client_example.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not Path(image_path).exists():
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)
    
    # Initialize client
    client = DeepfakeDetectionClient()
    
    # Check health
    print("🏥 Checking API health...")
    health = client.health_check()
    print(f"   Status: {health['status']}")
    print(f"   Model: {health.get('model_loaded', 'Unknown')}")
    print()
    
    # Detect deepfake
    print(f"🔍 Analyzing image: {image_path}")
    result = client.detect_deepfake(image_path, include_visualization=False)
    
    print("\n📊 Results:")
    print(f"   Authentic: {result['is_authentic']}")
    print(f"   Label: {result['label']}")
    print(f"   Confidence: {result['confidence']:.2%}")
    print(f"   Probability Fake: {result['probability_fake']:.2%}")
    print(f"   Probability Real: {result['probability_real']:.2%}")
    print(f"   Processing Time: {result['processing_time_ms']:.2f}ms")
    print()
    
    # Interpretation
    if result['is_authentic']:
        print("✅ Image appears to be AUTHENTIC")
    else:
        print("⚠️  Image appears to be a DEEPFAKE")
    
    if result['confidence'] < 0.7:
        print("⚠️  Low confidence - manual review recommended")


if __name__ == "__main__":
    main()
