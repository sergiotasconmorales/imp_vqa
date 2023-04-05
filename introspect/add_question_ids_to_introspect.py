import json
from tqdm import tqdm
from os.path import join as jp

path_introspect = '../data/vqaintrospect'
base_filename = 'VQAIntrospect_<>v1.0.json'
subsets = ['train', 'val']

cnt = 1 # counter for ids
for s in subsets:
    with open(jp(path_introspect, base_filename.replace('<>', s)), 'r') as f:
        data = json.load(f)
    # for loop to go through all sub-questions, adding new subquestion_id
    for question_id, contents in tqdm(data.items()):
        introspect_entries = contents['introspect']
        if len(introspect_entries) == 0:
            continue
        for ie in introspect_entries:
            sub_qa_entries = ie['sub_qa']
            if len(sub_qa_entries) == 0:
                continue
            for sqae in sub_qa_entries:
                sqae['subquestion_id'] = int(str(cnt) + '0'*9) # insert id in new field for current sub-question
                cnt += 1
    # when finished, save file for current subset
    with open(jp(path_introspect, base_filename.replace('<>', s + '_withid')), 'w') as f:
        json.dump(data, f)