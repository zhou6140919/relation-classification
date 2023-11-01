import os
import json
import time
import wandb
import torch
from argparse import ArgumentParser, Namespace
from datasets import load_from_disk
from transformers import BertTokenizer
from model import BertForRelationClassification
from torch.utils.data import DataLoader
import numpy as np
from itertools import permutations
import logging
from tqdm import tqdm


timestamp = time.strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join("checkpoints", timestamp)
os.makedirs(output_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]',
                    handlers=[logging.FileHandler(os.path.join(output_dir, "train.log")), logging.StreamHandler()])
logger = logging.getLogger(__name__)
NUM_LABELS = 30

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
logger.info(config)
wandb.init(project="relation-classification", name=timestamp, config=config)

# Load dataset
dataset = load_from_disk(config.dataset_dir)
logger.info(len(dataset["train"]))
logger.info(len(dataset["dev"]))
logger.info(len(dataset["test"]))

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained(config.model_name)
model = BertForRelationClassification(config.model_name, NUM_LABELS)

# tokenizer.add_special_tokens(
#    {"additional_special_tokens": ["<ent>", "</ent>"]})
# model.resize_token_embeddings(len(tokenizer) + 2)
# ent_start_token_id = tokenizer.convert_tokens_to_ids('<ent>')
# ent_end_token_id = tokenizer.convert_tokens_to_ids('</ent>')


# def add_entities(sent, tup):
#    entities = set()
#    tup_list = tup.split(" | ")
#    for etp in tup_list:
#        e_list = etp.split(" ; ")
#        entities.add(e_list[0])
#        entities.add(e_list[1])
#    for e in entities:
#        sent = sent.replace(e, f"<ent>{e}</ent>")
#    return sent


# Tokenize dataset
def tokenize_function(examples):
    # examples["sent"] = [add_entities(sent, tup) for sent,
    #                    tup in zip(examples["sent"], examples["tup"])]
    inputs = tokenizer(
        examples["sent"], padding="max_length", max_length=config.max_seq_length)

    labels = []
    entity_positions = []
    for sent, tup in zip(inputs['input_ids'], examples['tup']):
        entities = set()
        gold_pairs = []
        gold_labels = []
        tup_list = tup.split(" | ")
        for etp in tup_list:
            e_list = etp.split(" ; ")
            gold_pairs.append((e_list[0], e_list[1]))
            gold_labels.append(e_list[2])
            entities.add(e_list[0])
            entities.add(e_list[1])

        token_ids = sent
        entities = list(entities)
        tok_entities = [tokenizer.tokenize(e) for e in entities]
        entity_indices = []

        for entity in tok_entities:
            entity_len = len(entity)
            for i in range(len(token_ids) - entity_len + 1):
                if token_ids[i:i+entity_len] == [tokenizer.convert_tokens_to_ids(e) for e in entity]:
                    entity_indices.append((i, i+entity_len))
                    break

        one_sent_labels = []
        all_entity_pairs = list(permutations(entities, 2))
        one_sent_entity_indices = list(permutations(entity_indices, 2))
        for i, (e1, e2) in enumerate(all_entity_pairs):
            tmp_labels = [0] * NUM_LABELS
            while (e1, e2) in gold_pairs:
                tmp_labels[int(gold_labels[gold_pairs.index((e1, e2))])] = 1
                gold_labels.remove(gold_labels[gold_pairs.index((e1, e2))])
                gold_pairs.remove((e1, e2))
            one_sent_labels.append(tmp_labels)
        labels.append(one_sent_labels)
        entity_positions.append(one_sent_entity_indices)
    inputs["labels"] = labels
    inputs["entity_positions"] = entity_positions

    return inputs


tokenized_datasets = dataset.map(tokenize_function, batched=True)
# logger.info(tokenized_datasets.column_names)
tokenized_datasets.remove_columns(["sent", "tup"])


def collate_batch(batch):
    # Your custom logic here
    return {
        "input_ids": torch.tensor([item["input_ids"] for item in batch]),
        "labels": [item["labels"] for item in batch],
        "attention_mask": torch.tensor([item["attention_mask"] for item in batch]),
        "token_type_ids": torch.tensor([item["token_type_ids"] for item in batch]),
        "entity_positions": [item["entity_positions"] for item in batch]
    }


# small_train_dataset = tokenized_datasets["train"].shuffle(
#    seed=42).select(range(1000))
# small_eval_dataset = tokenized_datasets["test"].shuffle(
#    seed=42).select(range(1000))

# train_loader = DataLoader(
#    small_train_dataset, batch_size=config.train_batch_size, shuffle=True, collate_fn=collate_batch)
# eval_loader = DataLoader(
#    small_eval_dataset, batch_size=config.eval_batch_size, shuffle=False, collate_fn=collate_batch)

train_loader = DataLoader(
    tokenized_datasets["train"], batch_size=config.train_batch_size, shuffle=True, collate_fn=collate_batch)
eval_loader = DataLoader(
    tokenized_datasets["dev"], batch_size=config.eval_batch_size, shuffle=False, collate_fn=collate_batch)
test_loader = DataLoader(
    tokenized_datasets["test"], batch_size=config.eval_batch_size, shuffle=False, collate_fn=collate_batch)


def save_div(a, b):
    if b == 0:
        return 0
    else:
        return a / b


def compute_f1(predictions, labels):
    pred_count = 0
    gold_count = 0
    right_count = 0
    for one_sent_predicted_labels, one_sent_labels in zip(predictions, labels):
        for predicted_label, label in zip(one_sent_predicted_labels, one_sent_labels):
            pred_count += predicted_label.count(1)
            gold_count += label.count(1)
            for p, l in zip(predicted_label, label):
                if p == 1 and l == 1:
                    right_count += 1
    precision = save_div(right_count, pred_count)
    recall = save_div(right_count, gold_count)
    f1 = save_div(2 * precision * recall, precision + recall)
    return {'eval_f1': f1}


optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)


# Training
best_epoch, best_f1 = 0, 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
for epoch in range(config.num_train_epochs):
    logger.info(f"Epoch: {epoch}")
    logger.info("Start training")
    model.train()
    pbar = tqdm(train_loader, total=len(train_loader), dynamic_ncols=True)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"]
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        entity_positions = batch["entity_positions"]
        outputs = model(input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids, entity_positions=entity_positions, labels=labels)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        pbar.set_postfix({"loss": loss.item()})
        wandb.log({"train_loss": loss.item()})
    logger.info("Start evaluation")
    model.eval()
    pbar.close()
    all_predicted_labels = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(eval_loader, total=len(eval_loader), dynamic_ncols=True):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"]
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            entity_positions = batch["entity_positions"]
            outputs = model(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, entity_positions=entity_positions, labels=labels)
            loss = outputs[0]
            predicted_labels = outputs[1]
            all_predicted_labels.extend(predicted_labels)
            all_labels.extend(labels)

        f1 = compute_f1(all_predicted_labels, all_labels)
        logger.info(
            f"Epoch: {epoch}, eval_f1: {f1['eval_f1']}")
        wandb.log({"eval_loss": loss.item(), "eval_f1": f1["eval_f1"]})
        if f1["eval_f1"] > best_f1:
            best_f1 = f1["eval_f1"]
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(
                output_dir, "best_model.pt"))
    logger.info(f"Best epoch: {best_epoch}, best_f1: {best_f1}")

# Test
model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pt")))
model.eval()
pbar = tqdm(test_loader, total=len(test_loader), dynamic_ncols=True)
all_predicted_labels = []
all_labels = []
with torch.no_grad():
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"]
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        entity_positions = batch["entity_positions"]
        outputs = model(input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids, entity_positions=entity_positions, labels=labels)
        loss = outputs[0]
        predicted_labels = outputs[1]
        all_predicted_labels.extend(predicted_labels)
        all_labels.extend(labels)
    f1 = compute_f1(all_predicted_labels, all_labels)
    logger.info(f"test_f1: {f1['eval_f1']}")
    wandb.log({"test_f1": f1["eval_f1"]})
