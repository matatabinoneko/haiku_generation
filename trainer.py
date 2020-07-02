from LanguageModel import LM
import os
import torch
import time
import torch.nn.functional as F
from Vocabulary import Vocabulary
import math


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class LanguageModelTrainer:
    def __init__(
            self,
            model,
            optimizer,
            data_loader,
            device,
            max_epochs,
            log_interval=None,
            clip=0.25):  # 勾配をクリッピング

        self.model = model
        self.optimizer = optimizer
        self.model.to(device)
        self.device = device
        self.data_loader = data_loader
        self.max_epochs = max_epochs
        self.log_interval = log_interval
        self.clip = clip
        self.model_dir = './model'

    def train(self):
        print('Run trainer')

        pad = self.model.vocab.get_index('<pad>')

        start_at = time.time()

        for epoch in range(self.max_epochs):
            loss_epoch = 0.
            num_token = 0
            step = 0
            for batch_count, batch in enumerate(self.data_loader):
                state = None

                batch = batch.to(self.device)
                step += 1

                # 一個違いにしてinput[i]の予測がtarget[i]になるようにしている
                input_i = batch[:, :-1]
                target_i = batch[:, 1:]

                x, (h, c) = self.model(input_i, state)

                vocab_size = x.size(2)
                num_token_i = (target_i != pad).sum().item()  # 文章の長さを取得する
                # contiguousでメモリ上で要素順に並べる（transposeなどで順番が崩れたとき用？）
                # contiguous().view()はreshape()と等価
                loss = F.nll_loss(
                    F.log_softmax(
                        x, dim=-1).contiguous().view(-1, vocab_size),
                    target_i.contiguous().view(-1),
                    # reduction='sum',損失は単語ごとの平均にしたいが，padは無視したいのでここではsumにして後々平均を計算
                    ignore_index=pad)  # 損失計算に無視される

                self.optimizer.zero_grad()

                loss.div(num_token_i).backward()  # 文字数で平均する

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.clip)

                self.optimizer.step()
                loss_epoch += loss.item()

                h = h.clone().detach()
                c = c.clone().detach()
                state = (h, c)

                elapsed = time.time() - start_at
                # print(f'epoch:{epoch} step:{step}'
                #         f' loss:{loss_epoch/num_token:.2f}'
                #         f' elapsed:{elapsed:.2f}')

            loss_epoch /= batch_count
            ppl = math.exp(loss_epoch)
            elapsed = time.time() - start_at
            print('-' * 50)
            print(f'epoch:{epoch} loss:{loss_epoch:.2f}'
                  f' ppl:{ppl:.2f} elapsed:{elapsed:.2f}')
            decoded = self.model.generate(start=None)
            print(f'Sampled: {decoded}')
            print('-' * 50)
            if (epoch+1) % 10 == 0:
                if not os.path.exists(self.model_dir):
                    os.makedirs(self.model_dir)
                torch.save(
                    self.model, '{}/epoch{}.model'.format(self.model_dir, epoch+1))


def run_trainer(
        dim_emb,
        dim_hid,
        data_path,
        batch_size,
        max_epochs,
        device='cpu',
):
    data_loader = torch.utils.data.DataLoader(
        MyDataset(data_path=data_path), batch_size=batch_size, shuffle=True)
    print('data loaded from {} : {}lines'.format(data_path, len(data_loader)))

    model = LM(dim_emb=dim_emb, dim_hid=dim_hid)
    print(model)
    optimizer = torch.optim.Adam(model.parameters())
    # optimizer = torch.optim.Adam(
    #     model.parameters(), lr=0.005, betas=(0.9, 0.98), weight_decay=0.0001)

    trainer = LanguageModelTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        max_epochs=max_epochs,
        data_loader=data_loader
    )
    trainer.train()
