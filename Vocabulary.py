import collections
import math
import os
import random

import torch
import torch.nn.functional as F
import logging
from glob import glob
from itertools import chain
from collections import Counter


class Vocabulary:
    def __init__(self):
        self.index2word = {}
        self.word2index = {}
        self.add_word('<unk>')
        self.add_word('<pad>')
        self.add_word('<eos>')
        self.add_word('<bos>')
        self.add_word('<sep>')

    def __len__(self):
        return len(self.word2index)

    #　 if 'hoge' in Vocabulary　とかすると，containsが返される
    def __contains__(self, word):
        return word in self.word2index.keys()

    def add_word(self, word):
        if word not in self.word2index.keys():
            self.index2word[len(self.index2word)] = word
            self.word2index[word] = len(self.word2index)

    def get_word(self, index):
        return self.index2word[index]

    def get_index(self, word):
        return self.word2index[word]

    def sentence2index(self, wakati, length):
        ret = [self.get_index(word) if word in self.word2index.keys(
        ) else self.get_index('<unk>')for word in wakati]
        pad = [self.get_index('<pad>') for i in range(length-len(wakati))]
        eos = [self.get_index('<eos>')]
        bos = [self.get_index('<bos>')]
        if len(ret) <= length:
            return bos+ret+eos+pad
        else:
            return bos+ret[:length]+eos

    def save(self, vocab_file):
        with open(vocab_file, 'w') as f:
            for word in self.word2index:
                print(word, file=f)

    def load(self, vocab_file):
        with open(vocab_file, 'r') as f:
            for line in f:
                line = line.rstrip()
                self.add_word(line)
        print("load from {}\t word size:{}".format(
            vocab_file, len(self.word2index)))

    def load_word_from_data(self, train_path, dev_path):
        def count(f):
            for line in f:
                line = line.rstrip().split()
                for word in line:
                    yield word
        with open(train_path, 'r') as f_t, open(dev_path, 'r') as f_d:
            counter = Counter(count(chain(f_t, f_d)))
            for word, cnt in counter.most_common():
                self.add_word(word)
        # self.save(vocab_file='vocab_file')
