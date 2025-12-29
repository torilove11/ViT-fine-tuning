from datasets import load_dataset
from transformers import ViTImageProcessor
from src.config import model_name, dataset_name
from torchvision.transforms import (
    Compose, RandomResizedCrop, RandomHorizontalFlip, 
    ToTensor, Normalize, Resize, CenterCrop,
    RandomRotation, ColorJitter
)

def make_dataset():
    ds = load_dataset("maurice-fp/stanford-dogs")
    processor = ViTImageProcessor.from_pretrained(model_name)
    normalize = Normalize(mean=processor.image_mean, std=processor.image_std)

    # 학습용 transform 정의 (데이터 증강)
    _train_transforms = Compose([
        RandomResizedCrop(224, scale=(0.5, 1.0)), # 이미지를 랜덤하게 자르고 224로 맞춤
        RandomHorizontalFlip(), # 50% 확률로 좌우 반전
        RandomRotation(degrees=15), # 고개 기울임
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # 조명/털색
        ToTensor(),
        normalize,
    ])

    # 검증용 transform 정의 (증강 X)
    _val_transforms = Compose([
        Resize(256),
        CenterCrop(224),       
        ToTensor(),
        normalize,
    ])

    def train_transforms(examples):
        examples['pixel_values'] = [
            _train_transforms(image.convert("RGB")) for image in examples['image']
        ]
        return examples

    def val_transforms(examples):
        examples['pixel_values'] = [
            _val_transforms(image.convert("RGB")) for image in examples['image']
        ]
        return examples
    
    ds['train'].set_transform(train_transforms)
    ds['test'].set_transform(val_transforms)

    train_ds = ds['train']
    val_ds = ds['test']
    labels = ds['train'].features['label'].names

    return train_ds, val_ds, labels, processor