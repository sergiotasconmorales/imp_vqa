import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pickle
import os
from tqdm import tqdm
from transformers import BertTokenizer

class MNLIDataBert(Dataset):

  def __init__(self, train_df, val_df):
    self.label_dict = {'entailment': 0, 'contradiction': 1, 'neutral': 2}

    self.train_df = train_df
    self.val_df = val_df

    self.base_path = '/content/'
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    self.train_data = None
    self.val_data = None
    self.init_data()

  def init_data(self):
    # Saving takes too much RAM
    #
    # if os.path.exists(os.path.join(self.base_path, 'train_data.pkl')):
    #   print("Found training data")
    #   with open(os.path.join(self.base_path, 'train_data.pkl'), 'rb') as f:
    #     self.train_data = pickle.load(f)
    # else:
    #   self.train_data = self.load_data(self.train_df)
    #   with open(os.path.join(self.base_path, 'train_data.pkl'), 'wb') as f:
    #     pickle.dump(self.train_data, f)
    # if os.path.exists(os.path.join(self.base_path, 'val_data.pkl')):
    #   print("Found val data")
    #   with open(os.path.join(self.base_path, 'val_data.pkl'), 'rb') as f:
    #     self.val_data = pickle.load(f)
    # else:
    #   self.val_data = self.load_data(self.val_df)
    #   with open(os.path.join(self.base_path, 'val_data.pkl'), 'wb') as f:
    #     pickle.dump(self.val_data, f)
    self.train_data = self.load_data(self.train_df)
    self.val_data = self.load_data(self.val_df)

  def load_data(self, df):
    MAX_LEN = 512
    token_ids = []
    mask_ids = []
    seg_ids = []
    y = []

    premise_list = df['sentence1'].to_list()
    hypothesis_list = df['sentence2'].to_list()
    label_list = df['gold_label'].to_list()

    print('Loading data...')
    for (premise, hypothesis, label) in tqdm(zip(premise_list, hypothesis_list, label_list), total=len(premise_list)):
      if label not in list(self.label_dict): 
          continue # some label in validation is '-', so skip it
      premise_id = self.tokenizer.encode(premise, add_special_tokens = False)
      hypothesis_id = self.tokenizer.encode(hypothesis, add_special_tokens = False)
      pair_token_ids = [self.tokenizer.cls_token_id] + premise_id + [self.tokenizer.sep_token_id] + hypothesis_id + [self.tokenizer.sep_token_id]
      premise_len = len(premise_id)
      hypothesis_len = len(hypothesis_id)

      segment_ids = torch.tensor([0] * (premise_len + 2) + [1] * (hypothesis_len + 1))  # sentence 0 and sentence 1
      attention_mask_ids = torch.tensor([1] * (premise_len + hypothesis_len + 3))  # mask padded values

      token_ids.append(torch.tensor(pair_token_ids))
      seg_ids.append(segment_ids)
      mask_ids.append(attention_mask_ids)
      y.append(self.label_dict[label])
    
    token_ids = pad_sequence(token_ids, batch_first=True)
    mask_ids = pad_sequence(mask_ids, batch_first=True)
    seg_ids = pad_sequence(seg_ids, batch_first=True)
    y = torch.tensor(y)
    dataset = TensorDataset(token_ids, mask_ids, seg_ids, y)
    print(len(dataset))
    return dataset

  def get_data_loaders(self, batch_size=32, shuffle=True):
    train_loader = DataLoader(
      self.train_data,
      shuffle=shuffle,
      batch_size=batch_size
    )

    val_loader = DataLoader(
      self.val_data,
      shuffle=shuffle,
      batch_size=batch_size
    )

    return train_loader, val_loader


class TestDataBert(Dataset):
  def __init__(self, test_df):
    self.test_df = test_df

    self.base_path = '/content/'
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    self.test_data = self.preprocess()

  def preprocess(self):
    # function to pre-process the sentences in the same way they were processed in the class above
    MAX_LEN = 512
    token_ids = []
    mask_ids = []
    seg_ids = []
    premise_list = self.test_df['sentence1'].to_list()
    hypothesis_list = self.test_df['sentence2'].to_list()

    print('Loading data...')
    for (premise, hypothesis) in tqdm(zip(premise_list, hypothesis_list), total=len(premise_list)):
      premise_id = self.tokenizer.encode(premise, add_special_tokens = False)
      hypothesis_id = self.tokenizer.encode(hypothesis, add_special_tokens = False) 
      pair_token_ids = [self.tokenizer.cls_token_id] + premise_id + [self.tokenizer.sep_token_id] + hypothesis_id + [self.tokenizer.sep_token_id]
      premise_len = len(premise_id)
      hypothesis_len = len(hypothesis_id)     
      segment_ids = torch.tensor([0] * (premise_len + 2) + [1] * (hypothesis_len + 1))  # sentence 0 and sentence 1
      attention_mask_ids = torch.tensor([1] * (premise_len + hypothesis_len + 3))  # mask padded values
      token_ids.append(torch.tensor(pair_token_ids))
      seg_ids.append(segment_ids)
      mask_ids.append(attention_mask_ids)

    token_ids = pad_sequence(token_ids, batch_first=True)
    mask_ids = pad_sequence(mask_ids, batch_first=True)
    seg_ids = pad_sequence(seg_ids, batch_first=True)
    dataset = TensorDataset(token_ids, mask_ids, seg_ids)
    return dataset


  def get_data_loaders(self, batch_size=32):
    return DataLoader(
                        self.test_data,
                        shuffle=False,
                        batch_size=batch_size
                      )