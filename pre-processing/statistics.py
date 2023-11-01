"""
Observe the token length of each text in the dataset.
"""
import pandas as pd
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer

dataset = load_from_disk("datasets/nyt")
print(len(dataset["train"]))

tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")

token_lens = []
for i, example in tqdm(enumerate(dataset["train"]), total=len(dataset["train"])):
    tokens = tokenizer.encode(example['tup'], max_length=512, truncation=True)
    token_lens.append(len(tokens))


# show how many sentences are longer than 512 tokens
# print(len([i for i in token_lens if i >= 512]))
# use df.describe()
print(pd.Series(token_lens).describe())
