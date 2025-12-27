from datasets import load_dataset
from transformers import ViTImageProcessor
from src.config import model_name, dataset_name

def make_dataset():
    ds = load_dataset("maurice-fp/stanford-dogs")
    processor = ViTImageProcessor.from_pretrained(model_name)

    def transform(example_batch):
        images = [x.convert("RGB") for x in example_batch['image']]
        inputs = processor([x.convert("RGB") for x in example_batch['image']], return_tensors='pt')
        inputs['label'] = example_batch['label']
        return inputs
    
    prepared_ds = ds.with_transform(transform)

    train_ds = prepared_ds['train']
    val_ds = prepared_ds['test']
    labels = ds['train'].features['label'].names

    return train_ds, val_ds, labels, processor