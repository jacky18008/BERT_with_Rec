import transformers
from model import IEBPR
from dataset import Triplet_Dataset
import argparse
from run import train, eval
import json
from torch import nn, optim
from transformers import BertConfig, BertModel
from transformers import DistilBertConfig, DistilBertModel
from tqdm import tqdm
import torch

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--uiw_path')
parser.add_argument('--itext_path')
parser.add_argument('--config_path')
parser.add_argument('--eval_only', type=bool, default=False)
args = parser.parse_args()

with open(args.config_path) as fp:
    config = json.load(fp)
encoder_config = None
train_dataset = Triplet_Dataset(args.uiw_path+".train", args.itext_path, config)
test_dataset = Triplet_Dataset(args.uiw_path+".test", args.itext_path, config)
if config["encoder"].lower() == "distilbert":
    encoder_config = DistilBertConfig()
    pre_trained_model = 'distilbert-base-uncased'
elif config["encoder"].lower() == "bert":
    encoder_config = BertConfig()
    pre_trained_model = 'bert-base-uncased'
device = config["device"]

if config["pre_trained_rec_model_path"] != "":
    try:
        model = torch.load(config["pre_trained_rec_model_path"])
    except: 
        model = IEBPR.from_pretrained(pre_trained_model, config=encoder_config, model_config=config, dataset=train_dataset).to(device)
else:
    model = IEBPR.from_pretrained(pre_trained_model, config=encoder_config, model_config=config, dataset=train_dataset).to(device)
model.set_item_embed_table()
# test_meta = model.tokenizer('aaaaa', return_tensors='pt', padding=True, truncation=True)
# print(f"test tokenizer: {test_meta}")
# print(f"test tokenizer: {model.item_encode(**test_meta)}")
loss_fcn = model.bpr_loss
optimizer = optim.Adam(model.parameters(), lr=config["lr"])
weight_decay = config["reg_weight"]
'''
# print("model.parameters(): ")
l1 = []
for p in model.parameters():
    if p.requires_grad:
        l1.append(p)
bert_model = BertModel.from_pretrained('distilbert-base-uncased')
l2 = []
for p in bert_model.parameters():
    if p.requires_grad:
        l2.append(p)
assert len(l1) > len(l2)
'''

if __name__ == '__main__':
    if not args.eval_only:
        train_epochs = config["train_epochs"]
        for epoch in tqdm(range(1, train_epochs+1)):
            train(train_dataset, model, loss_fcn, optimizer, weight_decay, epoch, device=device)
            if epoch % 10 == 0:
                torch.save(model, f"model.epoch-{train_epochs}.pt")
                results = eval(test_dataset, model, train_dataset.allPos, config)
                print(f"eval result at {epoch} run: ", results)   
    results = eval(test_dataset, model, train_dataset.allPos, config)
    print("eval result: ", results)
