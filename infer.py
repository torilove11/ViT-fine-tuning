import os
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
from src.config import output_dir

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, output_dir)

model = ViTForImageClassification.from_pretrained(MODEL_PATH)
processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
model.eval()

def predict(image: Image.Image):
    image = image.convert("RGB")
    inputs = processor(image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze()
    
    pred_id = probs.argmax().item()
    raw_label = model.config.id2label[pred_id]
    pred_label = raw_label.split('-')[-1]
    confidence = probs[pred_id].item()
    
    return pred_label, confidence