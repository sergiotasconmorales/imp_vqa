# Script to create trainval_ans2label.json and trainval_label_2ans.json using all answers from train and val

import json
from collections import Counter
from os.path import join as jp
from copy import deepcopy

path_data = './data/lxmert/data/introspect'
path_lxmert_ans = './data/lxmert'
min_freq = 5
unknown_token = 'UNK'

subsets = ['train', 'val']

# load lxmert answers
with open(jp(path_lxmert_ans, 'all_ans.json'), 'r') as f:
    answers_lxmert = json.load(f)

answers_lxmert = set([e['ans'] for e in answers_lxmert])

all_ans = []
for s in subsets:
    # read data
    with open(jp(path_data, '{}_unprocessed.json'.format(s))) as f:
        data = json.load(f)
    for e in data:
        for k,_ in e['label'].items():
            all_ans.append(k)

# count answers' frequency
counts = Counter(all_ans).most_common()
chosen_answers = [e[0] for e in counts if e[1]>=min_freq and e[0] in answers_lxmert]
# add UNK token
chosen_answers.append(unknown_token)

ans2label = {e:i for e,i in zip(chosen_answers, range(len(chosen_answers)))}
label2ans = deepcopy(chosen_answers)

# save dicts
with open(jp(path_data, 'trainval_ans2label.json'), 'w') as f:
    json.dump(ans2label, f)

with open(jp(path_data, 'trainval_label2ans.json'), 'w') as f:
    json.dump(label2ans, f)

# modify answers in dataset and replace answers that are not in chosen_answers
for s in subsets:
    # read data
    with open(jp(path_data, '{}_unprocessed.json'.format(s))) as f:
        data = json.load(f)
    for e in data:
        # add some rules for typos
        if 'yea' in e['label']:
            p = e['label'].pop('yea')
            e['label']['yes'] = p

        lab_copy = deepcopy(e['label'])
        for k,_ in e['label'].items():
            if k not in chosen_answers:
                value = lab_copy.pop(k)
                lab_copy['UNK'] = value
        e['label'] = lab_copy
    # save
    with open(jp(path_data, '{}.json'.format(s)), 'w') as f:
        json.dump(data, f)