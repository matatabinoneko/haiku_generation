import torch
import torch.nn as nn
import torch.optim as optim
import random
from Vocabulary import Vocabulary

# 諸々のパラメータなど
# dim_emb = 200
# dim_hid = 128
# BATCH_NUM = 100
# vocab_size = len(char2id)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Encoderクラス


class Encoder_Decoder(nn.Module):
    def __init__(self,  dim_emb, dim_hid, vocab_file='./data/preprocessed/vocab_file.vocab'):
        super(Encoder_Decoder,self).__init__()
        self.vocab = Vocabulary()
        self.vocab.load(vocab_file=vocab_file)
        self.dim_hid = dim_hid
        self.word_embeddings = nn.Embedding(len(self.vocab), dim_emb)
        # self.gru = nn.GRU(dim_emb, dim_hid, batch_first=True)
        self.en_lstm = nn.LSTM(dim_emb, dim_hid, batch_first=True)

        self.de_lstm = nn.LSTM(dim_emb, dim_hid, batch_first=True)
        # LSTMの128次元の隠れ層を13次元に変換する全結合層
        self.hidden2linear = nn.Linear(dim_hid, len(self.vocab))

    def forward(self, sequence,state=None):
        embedding = self.word_embeddings(sequence)
        hs, (h,c) = self.en_lstm(embedding,state)

        output, (h,c) = self.de_lstm(embedding, (h,c))

        # アテンションを計算
        # t_output = torch.transpose(output, 1, 2)
        # s = torch.bmm(hs, t_output)
        # attention_weight = self.softmax(s)

        output = self.hidden2linear(output)
        return output, (h, c)

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
