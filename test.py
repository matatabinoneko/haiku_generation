from LanguageModel import LM
from trainer import run_trainer
import torch


# trainpath = './scraping'
batch_size = 32
# threshhold = 10
dim_emb = 512
dim_hid = 512
data_path = './data/preprocessed/train.interm.tokenized.preprocessed'


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(device)
run_trainer(device=device, dim_emb=dim_emb, dim_hid=dim_hid,
            max_epochs=1000, data_path=data_path, batch_size=batch_size)
