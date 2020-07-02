import json
import random
import argparse
import os
import torch
from Vocabulary import Vocabulary
from itertools import chain
from more_itertools import ilen
from tqdm import tqdm


ROOT_DIR = "./data/"
PREPROCESSED_DIR = ROOT_DIR+'preprocessed/'


def main(args):
    train_input_filepath = args.train_input
    dev_input_filepath = args.dev_input

    vocab = Vocabulary()
    vocab.load_word_from_data(
        train_path=train_input_filepath, dev_path=dev_input_filepath)
    vocab.save(vocab_file=PREPROCESSED_DIR+'vocab_file.vocab')
    print("vocab size:{}".format(len(vocab)))

#     max_length = -1
#     with open(train_input_filepath, 'r') as f_t, open(dev_input_filepath, 'r') as f_d:
#         for line in chain(f_t, f_d):
#             max_length = max(max_length, len(line.rstrip().split()))
    max_length = 17

    for file_name in [train_input_filepath, dev_input_filepath]:
        output_file = file_name.split('/')[-1] + '.preprocessed'
        data = []
        with open(file_name, 'r') as fin, open(PREPROCESSED_DIR+output_file, 'w') as fout:
            total_len = ilen(fin)
            bar = tqdm(total=total_len)
            fin.seek(0)
            for i, line in enumerate(fin, start=1):
                line = line.rstrip().split()
                data.append(torch.LongTensor(
                    vocab.sentence2index(line, max_length)))
                # data = ' '.join(
                #     list(map(str, vocab.sentence2index(line, max_length))))
                # if i == total_len:
                #     print(data, file=fout, end='')
                # else:
                #     print(data, file=fout)
                bar.update(1)
        torch.save(data, PREPROCESSED_DIR+output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--train_input', help='input file path')
    parser.add_argument(
        '-d', '--dev_input', help='input file path')
    args = parser.parse_args()

    if not os.path.isdir('./data'):
        os.makedirs('./data')
    if not os.path.isdir('./data/preprocessed'):
        os.mkdir('./data/preprocessed')
    main(args)
