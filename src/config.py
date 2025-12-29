from transformers import TrainingArguments

model_name = "google/vit-base-patch16-224-in21k"
output_dir = "./vit-stanford-dogs-finetuned"
dataset_name = "maurice-fp/stanford-dogs"

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=32,
    eval_strategy="steps",
    num_train_epochs=10,
    fp16=True,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to="tensorboard",
    load_best_model_at_end=True,

    metric_for_best_model="accuracy",
    greater_is_better=True,           
    label_smoothing_factor=0.1
)