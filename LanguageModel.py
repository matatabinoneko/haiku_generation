import random
import torch
import torch.nn.functional as F
from Vocabulary import Vocabulary
from bisect import bisect
from itertools import accumulate
import numpy as np

device = 'cuda:2'


class LM(torch.nn.Module):
    def __init__(self, dim_emb, dim_hid, vocab_file='./data/preprocessed/vocab_file.vocab'):
        super().__init__()

        self.vocab = Vocabulary()
        self.vocab.load(vocab_file=vocab_file)
        self.embed = torch.nn.Embedding(len(self.vocab), dim_emb)
        self.rnn1 = torch.nn.LSTM(dim_emb, dim_hid, batch_first=True)
        self.rnn2 = torch.nn.LSTM(dim_hid, dim_hid, batch_first=True)
#         self.rnn3 = torch.nn.LSTM(dim_hid, dim_hid, batch_first=True)
#         self.rnn4 = torch.nn.LSTM(dim_hid, dim_hid, batch_first=True)
        self.out = torch.nn.Linear(dim_hid, len(self.vocab))

    def forward(self, x, state1=None, state2=None):
        out = self.embed(x)
        out, state1 = self.rnn1(out, state1)
        out, state2 = self.rnn2(out, state2)
#         out, (h, c) = self.rnn3(out, None)
#         out, (h, c) = self.rnn4(out, None)
        out = self.out(out)
        return out, state1, state2

    # def to_int(self, a):
    #     if a == -float('inf'):
    #         return 0
    #     else:
    #         return int(1e9*a)

    def generate(self, prefix, max_len=30):
        cost = 0
        softmax = torch.nn.Softmax(dim=-1)
        start = '<bos>'

        idx = self.embed.weight.new_full(
            (1, 1),
            self.vocab.get_index(start),
            dtype=torch.long)
        decoded = [start]
        state1, state2 = None, None
        unk = self.vocab.get_index('<unk>')
        while decoded[-1] != '<eos>' and len(decoded) < max_len:
            x, state1, state2 = self.forward(idx, state1, state2)

            if 0 < len(prefix):
                word = prefix.pop()
                idx = self.vocab.get_index(word)

                idx = torch.tensor(idx).view(1, 1).to(device)
            else:
                x[:, :, unk] = -float('inf')
                x = softmax(x)
                # idx = torch.argmax(x, dim=-1)
                x = x.squeeze().to('cpu').detach().numpy()
                accum = list(accumulate(x))
                idx = bisect(accum, random.random()*accum[-1])
                # word = self.vocab.get_word(idx.item())
                cost += np.log2(x[idx])
                word = self.vocab.get_word(idx)
                idx = torch.tensor(idx).view(1, 1).to(device)

            decoded.append(word)
        cost /= len(decoded)
        return ' '.join(decoded), cost
