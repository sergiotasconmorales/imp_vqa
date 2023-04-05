# script to finetune a BERT model using data from Multi-NLI dataset or SNLI dataset
# Based on https://towardsdatascience.com/fine-tuning-pre-trained-transformer-models-for-sentence-entailment-d87caf9ec9db

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
dataset_name = 'snli' # or multinli
path_data = './data/{}_1.0/{}_1.0'.format(dataset_name, dataset_name)
path_model_save = './models/bert_te/{}'.format(dataset_name)
if not os.path.exists(path_model_save): # make sure folder exists to save final model
    os.makedirs(path_model_save, exist_ok=True)

# declare device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

# read data from multi-MLI dataset
# train data
trainset_name = '{}_1.0_train.jsonl'.format(dataset_name)
df_train = pd.read_json(jp(path_data, trainset_name), lines=True)
train_df = df_train[['sentence1', 'sentence2', 'gold_label']]
# val
if dataset_name == 'snli':
    valset_name = 'snli_1.0_dev.jsonl'
elif dataset_name == 'multinli':
    valset_name = 'multinli_1.0_dev_matched.jsonl'
else:
    raise ValueError('Unknown dataset')
df_val = pd.read_json(jp(path_data, valset_name), lines=True)
val_df = df_val[['sentence1', 'sentence2', 'gold_label']]

# drop NaNs
train_df = train_df.dropna()
val_df = val_df.dropna()

# remove empty stuff
train_df = train_df[(train_df['sentence1'].str.split().str.len() > 0) & (train_df['sentence2'].str.split().str.len() > 0)]
val_df = val_df[(val_df['sentence1'].str.split().str.len() > 0) & (val_df['sentence2'].str.split().str.len() > 0)]

mnli_dataset = MNLIDataBert(train_df, val_df)

train_loader, val_loader = mnli_dataset.get_data_loaders(batch_size=batch_size)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
model.to(device)

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