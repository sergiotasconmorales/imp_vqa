# Script to re-structure the Introspect dataset (previously annotated with relations between main and sub-questions).
# Two main changes:
#   - Introspect samples (main and corresponding sub) that are in minival set of LXMERT are removed for fairness. 
#     In total 1723 main questions have to be excluded along with their corresponding sub-questions
#   - Data is structured so that it matches the fields in the VQA data used to finetune LXMERT. Necessary fields are 
#     answer_type, img_id, label, question_id, question_type and sent. Additional fields like <role> (main or sub),
#     <parent> (pointer to main question) and <rel> (relation between main and sub) are added for consistency purposes.
#
# Final version of introspect (saved in <path_output>) should be okay to finetune LXMERT

from tqdm import tqdm
import json
from os.path import join as jp
import os
from answer_processing import preprocess_answer

path_data = './data'
path_introspect = jp(path_data, 'vqaintrospect')
path_vqa_lxmert = jp(path_data, 'lxmert/data/vqa')
path_output = jp(path_data, 'lxmert/data/introspect')

os.makedirs(path_output, exist_ok=True)

subsets = ['train', 'val']

# for each subset
for s in subsets:
    new_introspect = []

    # step 1: data from VQA
    if s == 'train':
        with open(jp(path_vqa_lxmert, 'train.json'), 'r') as f:
            data_vqa = json.load(f)
    else: # read only nominival
        with open(jp(path_vqa_lxmert, 'nominival.json'), 'r') as f:
            data_vqa = json.load(f)

    # step 2: read introspect data
    with open(jp(path_introspect, 'VQAIntrospect_{}_no_duplicates_withidrelv1.0.json'.format(s)), 'r') as f:
        data_introspect = json.load(f)

    # step 3: if train, check that all introspect main questions are in the train set of VQA.
    #         if val, exclude samples that are in minival
    all_ids_vqa = {e['question_id'] for e in data_vqa}
    all_ids_introspect = set(map(int, list(data_introspect.keys()))) # converting ids to int
    if s == 'train':
        # Since VQAIntrospect uses samples from VQAv1 that are not in VQAv2, we have to remove those samples
        banned_train = all_ids_introspect - all_ids_vqa 
        for id_to_ignore in banned_train:
            data_introspect.pop(str(id_to_ignore))
    else:
        # exclude samples that are not in nominival (they are in minival)
        banned = all_ids_introspect - all_ids_vqa.intersection(all_ids_introspect) # must have 1723 elements [OK]
        for id_to_ignore in banned:
            data_introspect.pop(str(id_to_ignore))
        # len of data_introspect after removing banned elements: 12784 [OK]

    # step 4: generate new structure for introspect data (equalize format for question_id, img_id, etc)
    id2entry = {e['question_id']: e for e in data_vqa} # so that I can just copy the main question entry from the VQA data
    for k, v in tqdm(data_introspect.items()):
        # add main question first
        temp_dict = id2entry[int(k)]
        # add fields
        temp_dict['role'] = 'main'
        #* adding condition so that if main question's label is empty (many in official LXMERT VQA data) then exclude it from Introspect
        if len(temp_dict['label']) == 0:
            continue
        new_introspect.append(temp_dict)
        introspect_entries = v['introspect']
        if len(introspect_entries) == 0: # no sub-questions
            continue
        for ie in introspect_entries:
            sub_qa_entries = ie['sub_qa']
            if len(sub_qa_entries) == 0: # empty sub-question field
                continue
            for sqae in sub_qa_entries:
                temp_dict = {   'answer_type': 'other',
                                'img_id': 'COCO_{}2014_{}'.format(s, str(v['image_id']).zfill(12)), 
                                'label': {preprocess_answer(sqae['sub_answer']): 1},
                                'question_id': sqae['subquestion_id'],
                                'question_type': 'are', # assumed
                                'sent': sqae['sub_question'],
                                'role': 'sub',
                                'parent': int(k),
                                'rel': sqae['rel']}
                new_introspect.append(temp_dict)

# step 5: save new version of introspect
    print('Number of introspect samples (main + sub):', len(new_introspect))
    with open(jp(path_output, '{}_unprocessed.json'.format(s)), 'w') as f:
        json.dump(new_introspect, f)