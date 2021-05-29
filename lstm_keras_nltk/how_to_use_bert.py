from transformers import BertTokenizerFast
import torch
from rich.console import Console
import torch.nn.functional as F
from ast import literal_eval
from tqdm import trange
import json
import argparse

sentences = [
    "please insert a column inside this place",
    "give me a cup of tea",
    "glass of water",
    "pakora here is a tang cup liter",
    "table of plates",
    "marbles house",
]

tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
encoded = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sentences,
                                      padding=True,
                                      truncation=True)

lengths = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sentences,
                                      padding=False,
                                      truncation=True)

original = encoded["input_ids"]
L = [len(k) for k in lengths["input_ids"]]
