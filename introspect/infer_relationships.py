# Script to use re-finetuned model to predict the relationships in the remaining part of Introspect sentences that was not used for 
# the re-finetuning of the NLP model. 

import sys
sys.path.insert(0,'.')

import pandas as pd
import re
import torch
import json
import numpy as np
from os.path import join as jp
from bert.dataset.mli import TestDataBert
from transformers import BertForSequenceClassification
from bert.trainer import infer

version = 2 # version of annotated relationships dataset
dataset_name = 'introspectnli' # name of the NLI dataset that was used last time to (re)finetune the NLP model
for subset_to_predict in ['train', 'val']: # which (non-annotated) subset samples to predict the relationship for
    path_model = './models/bert_te_refinetune/{}'.format(dataset_name) # path to pre-trained weights
    path_sentences = './data/sentences_dataset'
    path_rels = './data/introspectnli_{}.0'.format(version)

    # now define the rules to go from NLI labels {entailment:0, contradiction:1, neutral:2} to relationships
    dict_rel_rules = {  (0,0): '<->', 
                        (0,2): '-->',
                        (0,1): '-->',
                        (2,0): '<--',
                        (2,2): '---',
                        (2,1): '---',
                        (1,0): '<--',
                        (1,2): '---',
                        (1,1): '-x-'}


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    # read some samples from sentences dataset
    with open(jp(path_sentences, '{}_sentence_pairs.json'.format(subset_to_predict)), 'r') as f:
        sentences = json.load(f)

    # open annotated relationships too
    with open(jp(path_rels, '{}_sentence_pairs_relationships.json'.format(subset_to_predict)), 'r') as f:
        relationships = json.load(f)

    sentence_pairs_to_ignore = relationships['indexes'] # indexes of sentence pairs that were already manually annotated

    # now  remove, from sentences, pairs with indexes in sentence_pairs_to_ignore
    to_infer_normal = sentences
    to_infer_normal = [sentences[i] for i in range(0, len(sentences)) if i not in sentence_pairs_to_ignore] # remove pairs that were already annotated
    # invert sentence1 and sentence2 so that model predicts both directions
    to_infer_inverted = [{'sentence1': e['sentence2'], 'sentence2': e['sentence1'], 'label':e['label'], 'image_id': e['image_id']} for e in to_infer_normal]
    to_infer_inverted = [sentences[i] for i in range(0, len(sentences)) if i not in sentence_pairs_to_ignore] # remove pairs that were already annotated

    # create df from sentences variable
    test_df_normal = pd.DataFrame(to_infer_normal)
    test_df_inverted = pd.DataFrame(to_infer_inverted)

    # now I need to create a dataset class for test, similar to the one used for training
    test_dataset_normal = TestDataBert(test_df_normal)
    test_dataset_inverted = TestDataBert(test_df_inverted)
    test_loader_normal = test_dataset_normal.get_data_loaders(batch_size=16)
    test_loader_inverted = test_dataset_inverted.get_data_loaders(batch_size=16)

    # create model 
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
    model.to(device)
    # load weights from path_model
    model_info = torch.load(jp(path_model, 'best_model.pt'), map_location=device)
    print('Loading weights from epoch', model_info['epoch'], 'corresponding to val loss', model_info['val_loss'])
    model.load_state_dict(model_info['model_state_dict'])

    predicted_labels_normal = infer(device, model, test_loader_normal, path_model, the_name='normal')
    assert len(to_infer_normal) == predicted_labels_normal.shape[0]

    predicted_labels_inverted = infer(device, model, test_loader_inverted, path_model, the_name='inverted')
    assert len(to_infer_inverted) == predicted_labels_inverted.shape[0]

    # now use predictions to get relationship between pairs
    idx = 0
    for e in to_infer_normal:
        pred_normal = predicted_labels_normal[idx].item()
        pred_inverted = predicted_labels_inverted[idx].item()
        # decide relationship depending on predictions
        rel = dict_rel_rules[(pred_normal, pred_inverted)]
        e['predicted'] = rel
        idx+=1

    with open(jp(path_sentences, subset_to_predict + '_predicted_relationships.json'), 'w') as f:
        json.dump(to_infer_normal, f)