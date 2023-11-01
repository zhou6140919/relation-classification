import os
import json
import time
import torch
from argparse import ArgumentParser, Namespace
from datasets import load_from_disk
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import evaluate

timestamp = time.strftime("%Y%m%d_%H%M%S")

# Load config
parser = ArgumentParser()
parser.add_argument(
    "--config",
    "-c",
    type=str,
    required=True,
    default="config/config.json",
    help="Path to config file"
)
args = parser.parse_args()
config = Namespace(
    **vars(args), **vars(Namespace(**json.load(open(args.config)))))
print(config)

# Load dataset
dataset = load_from_disk(config.dataset_dir)
print(len(dataset["train"]))
print(len(dataset["dev"]))
print(len(dataset["test"]))

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
if config.add_other:
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name, num_labels=30)
else:
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name, num_labels=29)

tokenizer.add_tokens(["<e1>", "</e1>", "<e2>", "</e2>"])
# model.resize_token_embeddings(len(tokenizer))


# Tokenize dataset
def tokenize_function(examples):
    labels = examples["label"]
    inputs = tokenizer(
        examples["context"], padding="max_length", max_length=config.max_seq_length, truncation=True)
    inputs["labels"] = labels
    return inputs


tokenized_datasets = dataset.map(tokenize_function, batched=True)
# print(tokenized_datasets.column_names)
tokenized_datasets.remove_columns(["context", "label"])


# small_train_dataset = tokenized_datasets["train"].shuffle(
#     seed=42).select(range(1000))
# small_eval_dataset = tokenized_datasets["test"].shuffle(
#     seed=42).select(range(1000))


# def preprocess_logits_for_metrics(logits, labels):
#    """
#    Original Trainer may have a memory leak.
#    This is a workaround to avoid storing too many tensors that are not needed.
#    """
#    pred_ids = torch.argmax(logits[0], dim=-1)
#    return pred_ids, labels


# def parse_output(outputs):
#     new_outputs = []
#     for p in outputs:
#         tmp = []
#         t = p.split(" | ")
#         for i in t:
#             tmp.append(i.split(" ; "))
#         new_outputs.append(tuple(tmp))
#     return new_outputs


def save_div(a, b):
    if b == 0:
        return 0
    else:
        return a / b


# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     predictions = predictions[0]
#     predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
#     predictions = parse_output(predictions)
#     labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
#     labels = parse_output(labels)
#     pred_count = 0
#     gold_count = 0
#     right_count = 0
#     for i, j in zip(predictions, labels):
#         pred_count += len(i)
#         gold_count += len(j)
#         for p in i:
#             if p in j:
#                 right_count += 1
#     precision = save_div(right_count, pred_count)
#     recall = save_div(right_count, gold_count)
#     f1 = save_div(2 * precision * recall, precision + recall)
#     return {'f1': f1}

metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average="micro")


# Train
training_args = TrainingArguments(
    do_train=False,
    do_eval=False,
    output_dir=os.path.join(config.output_dir, timestamp),
    num_train_epochs=config.num_train_epochs,
    per_device_train_batch_size=config.train_batch_size,
    per_device_eval_batch_size=config.eval_batch_size,
    warmup_ratio=config.warmup_ratio,
    run_name=timestamp,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    metric_for_best_model="eval_f1",
    learning_rate=config.learning_rate,
    load_best_model_at_end=True,
    logging_strategy="steps",
    logging_steps=10,
    save_total_limit=2
)

# logging_dir=os.path.join(config.output_dir, timestamp, "logs")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["dev"],
    compute_metrics=compute_metrics,
)

test_scores = trainer.evaluate(
    eval_dataset=tokenized_datasets["test"],
)

print(test_scores)
