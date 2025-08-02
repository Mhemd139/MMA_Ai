import os
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import cv2
import numpy as np

class PunchClassifier:
    def __init__(self, model_path="models/punch-detection-model"):
        """Initialize the punch classifier with trained model"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🤖 Loading model from: {model_path}")
        self.model = AutoModelForImageClassification.from_pretrained(model_path)
        self.processor = AutoImageProcessor.from_pretrained(model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded successfully on {self.device}")
    
    def predict_punch(self, image_path):
        """Predict if a punch is blocked or landed"""
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions).item()
            confidence = predictions[0][predicted_class].item()
        
        result = "Blocked" if predicted_class == 0 else "Landed"
        return result, confidence
    
    def predict_from_array(self, image_array):
        """Predict from numpy array (for video processing)"""
        # Convert BGR to RGB
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        
        image = Image.fromarray(image_array)
        inputs = self.processor(image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions).item()
            confidence = predictions[0][predicted_class].item()
        
        result = "Blocked" if predicted_class == 0 else "Landed"
        return result, confidence

def test_model():
    """Test the model on sample images"""
    classifier = PunchClassifier()
    
    # Test on sample images
    test_images = [
        "data/blocked_punches/frame_00060.jpg",
        "data/landed_punches/frame_00153.jpg"
    ]
    
    print("\n🧪 Testing model predictions:")
    for img_path in test_images:
        if os.path.exists(img_path):
            result, confidence = classifier.predict_punch(img_path)
            print(f"   {img_path}: {result} (confidence: {confidence:.3f})")
        else:
            print(f"   ⚠️ File not found: {img_path}")

if __name__ == "__main__":
    test_model() 