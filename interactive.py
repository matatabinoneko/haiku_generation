import torch
from LanguageModel import LM
import copy

N = 100
INF = 1e8


def main():
    model = torch.load('./model/epoch20.model')
    # model.generate = generate
    while True:
        print('key: ', end='')
        key = input()
        print('prefix: ', end='')
        prefix = input()
        if prefix == '' and key == '':
            sep = []
        else:
            sep = ['[SEP]']
        input_sentence = list(key)+sep+list(prefix)
        input_sentence = input_sentence[::-1]
        cost = -1*INF
        output = ''
        for i in range(N):
            tmp_output, tmp_cost = model.generate(copy.copy(input_sentence))
            if cost < tmp_cost:
                cost = tmp_cost
                output = tmp_output

        print('output:{:.3f} {}'.format(cost, output))


if __name__ == '__main__':
    main()
