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


class MODEL(torch.nn.Module):
    def __init__(self, params):
        # super(MODEL, self)
        super().__init__()
        params = vars(params)
        print(json.dumps(params, indent=6))
        self.embedding = torch.nn.Embedding(params['vocab_sz'],
                                            params['embed_dim'])
        self.lstm = torch.nn.LSTM(input_size=params['embed_dim'],
                                  hidden_size=params['lstm_hidden'],
                                  batch_first=True,
                                  num_layers=params['lstm_layers'],
                                  )
        self.hidden1 = torch.nn.Linear(params['lstm_hidden'] * params['lstm_layers'],
                                       params['h1'],
                                       )
        self.hidden2 = torch.nn.Linear(params['h1'],
                                       params['h2'],
                                       )
        self.out = torch.nn.Linear(params['h2'],
                                   params['nC'],
                                   )
        self.l = params['lstm_layers']
        self.h = params['lstm_hidden']

    def forward(self, x, L=None):
        print(f"{'INPUT':<20}{'---->':<20}{x.shape}")
        out = self.embedding(x)
        out = torch.nn.utils.rnn.pack_padded_sequence(out, lengths=L, batch_first=True, enforce_sorted=False)
        print(f"{'AFTER EMBEDDING':<20}{'---->':<20}{type(out)}")
        O, (h, c) = self.lstm(out)  # O is dependent on seq_len which is variable
        O, _ = torch.nn.utils.rnn.pad_packed_sequence(O, batch_first=True)
        # O-->[batch, seq_len, num_directions * hidden_size]
        # h-->[num_layers * num_directions, batch, hidden_size]
        # c-->[num_layers * num_directions, batch, hidden_size]

        print(f"{'AFTER LSTM->O':<20}{'---->':<20}{O.shape}")
        print(f"{'AFTER LSTM->h':<20}{'---->':<20}{h.shape}")
        print(f"{'AFTER LSTM->c':<20}{'---->':<20}{c.shape}")
        h = h.permute(1, 0, 2)
        print(f"{'AFTER TRANSPOSE':<20}{'---->':<20}{h.shape}")
        h = h.reshape(-1, self.l * self.h)
        print(f"{'AFTER RESHAPING':<20}{'---->':<20}{h.shape}")
        out = F.relu(self.hidden1(h))
        print(f"{'AFTER HIDDEN-1':<20}{'---->':<20}{out.shape}")
        out = F.relu(self.hidden2(out))
        print(f"{'AFTER HIDDEN-2':<20}{'---->':<20}{out.shape}")
        predictions = F.softmax(self.out(out))
        print(f"{'LAST LAYER':<20}{'---->':<20}{predictions.shape}")

        return predictions


print = Console().print
vocab_sz = len(tokenizer.get_vocab())

m = MODEL(argparse.Namespace(vocab_sz=vocab_sz,
                             embed_dim=512,
                             lstm_hidden=128,
                             lstm_layers=3,
                             h1=1024,
                             h2=256,
                             nC=5,
                             ))
import numpy as np
labels = np.random.choice(list(range(5)), (6,)) # batch-sz ====> 6, nC----> classes
labels = torch.from_numpy(labels).type(torch.LongTensor)
opt = torch.optim.Adam(params=m.parameters(), lr=1e-3)
for epoch in trange(100, desc="EPOCH"):
    outs = m(torch.tensor(original), L)
    loss = F.cross_entropy(outs, labels)
    print(f"LOSS ---> {loss}", style="color(58)")
    opt.zero_grad()
    loss.backward()
    opt.step()
