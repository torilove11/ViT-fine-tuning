import os
from transformers import Trainer, ViTForImageClassification
from src.config import output_dir, training_args
from src.data import make_dataset
from src.utils import compute_metrics, collate_fn

def main():

    print(f"[{output_dir}] 모델의 성능 평가를 시작합니다.")
    _, val_ds, labels, processor = make_dataset()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, output_dir)
    model = ViTForImageClassification.from_pretrained(MODEL_PATH)

    training_args.output_dir = MODEL_PATH
    training_args.report_to = []

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        eval_dataset=val_ds,
        processing_class=processor,
    )

    metrics = trainer.evaluate()

    print("\n" + "="*30)
    print("   Evaluation   ")
    print("="*30)
    print(f"Accuracy : {metrics['eval_accuracy']:.2%}")
    print(f"Loss     : {metrics['eval_loss']:.4f}")
    print(f"Inference Time: {metrics['eval_runtime']:.2f} sec")
    print("="*30)

if __name__ == "__main__":
    main()