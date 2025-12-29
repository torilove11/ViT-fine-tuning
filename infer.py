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
    
    topk_probs, topk_indices = torch.topk(probs, 3) # 상위 3개 추출

    results = {}
    for i in range(3):
        idx = topk_indices[i].item()
        score = topk_probs[i].item()
    
        raw_label = model.config.id2label[idx]
        clean_label = raw_label.split('-')[-1]
    
        results[clean_label] = score

    return results