# Script to re-finetune NLP model to new relationships. Previously fine-tuned model is loaded and then new data is used to finetune it.

import pandas as pd
import re
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, SequentialSampler, RandomSampler
from torch.nn.utils.rnn import pad_sequence
import pickle
import os
import numpy as np
from os.path import join as jp
from bert.dataset.mli import MNLIDataBert
from transformers import BertForSequenceClassification
from torch.optim import AdamW
from bert.trainer import train

# basic training params
batch_size = 16
epochs = 5
learning_rate = 2e-5

# path to data
version = 2
dataset_name = 'introspectnli'
path_model = './models/bert_te/snli'
path_data = './data/{}_{}.0'.format(dataset_name, str(version))
path_model_save = './models/bert_te_refinetune/{}'.format(dataset_name)
if not os.path.exists(path_model_save): # make sure folder exists to save final model
    os.makedirs(path_model_save, exist_ok=True)
    print('Folder created: {}'.format(path_model_save))

# declare device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

trainset_name = '{}_{}.0_train.jsonl'.format(dataset_name, str(version))
df_train = pd.read_json(jp(path_data, trainset_name), lines=True)
train_df = df_train[['sentence1', 'sentence2', 'gold_label']]

valset_name = '{}_{}.0_val.jsonl'.format(dataset_name, str(version))
df_val = pd.read_json(jp(path_data, valset_name), lines=True)
val_df = df_val[['sentence1', 'sentence2', 'gold_label']]

# drop NaNs
train_df = train_df.dropna()
val_df = val_df.dropna()

# remove empty stuff
train_df = train_df[(train_df['sentence1'].str.split().str.len() > 0) & (train_df['sentence2'].str.split().str.len() > 0)]
val_df = val_df[(val_df['sentence1'].str.split().str.len() > 0) & (val_df['sentence2'].str.split().str.len() > 0)]

nli_dataset = MNLIDataBert(train_df, val_df)

train_loader, val_loader = nli_dataset.get_data_loaders(batch_size=batch_size)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
model.to(device)

# load weights of first finetune
model_info = torch.load(jp(path_model, 'best_model.pt'), map_location=device)
print('Loading weights from epoch', model_info['epoch'], 'corresponding to val loss', model_info['val_loss'])
model.load_state_dict(model_info['model_state_dict'])

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')

train(epochs, device, model, train_loader, val_loader, optimizer, path_model_save)