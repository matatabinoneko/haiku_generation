import random
import torch
import torch.nn.functional as F
from Vocabulary import Vocabulary


class LM(torch.nn.Module):
    def __init__(self, dim_emb, dim_hid, vocab_file='./data/preprocessed/vocab_file.vocab'):
        super().__init__()

        self.vocab = Vocabulary()
        self.vocab.load(vocab_file=vocab_file)
        self.embed = torch.nn.Embedding(len(self.vocab), dim_emb)
        self.rnn1 = torch.nn.LSTM(dim_emb, dim_hid, batch_first=True)
#         self.rnn2 = torch.nn.LSTM(dim_hid, dim_hid, batch_first=True)
#         self.rnn3 = torch.nn.LSTM(dim_hid, dim_hid, batch_first=True)
#         self.rnn4 = torch.nn.LSTM(dim_hid, dim_hid, batch_first=True)
        self.out = torch.nn.Linear(dim_hid, len(self.vocab))

    def forward(self, x, state=None):
        x = self.embed(x)
        x, (h, c) = self.rnn1(x, state)
#         x, (h, c) = self.rnn2(x, None)
#         x, (h, c) = self.rnn3(x, None)
#         x, (h, c) = self.rnn4(x, None)
        x = self.out(x)
        return x, (h, c)

    # def to_int(self, a):
    #     if a == -float('inf'):
    #         return 0
    #     else:
    #         return int(1e9*a)

    def generate(self, start=None, max_len=17):

        if start is None:
            start = random.choice(self.vocab.index2word)

        idx = self.embed.weight.new_full(
            (1, 1),
            self.vocab.get_index(start),
            dtype=torch.long)
        decoded = [start]
        state = None
        unk = self.vocab.get_index('<unk>')
        while decoded[-1] != '<eos>' and len(decoded) < max_len:
            x, state = self.forward(idx, state)
            x[:, :, unk] = -float('inf')

            # prob = list(map(self.to_int, x.squeeze().tolist()))

            # idx = torch.tensor(random.choices(
            #     list(range(len(prob))), weights=prob, k=1)).view(1, -1)

            idx = torch.argmax(x, dim=-1)

            word = self.vocab.get_word(idx.item())
            decoded.append(word)
        return ' '.join(decoded)
