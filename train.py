import sys
from transformers import Trainer
from src.config import training_args, output_dir
from src.data import make_dataset
from src.model import get_model
from src.utils import compute_metrics, collate_fn

def main():
    train_ds, val_ds, labels, processor = make_dataset()
    model = get_model(labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=processor,
    )

    print("모델 학습을 시작합니다.")
    train_results = trainer.train()

    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()
    
    # eval
    metrics = trainer.evaluate(val_ds)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    main()