import json
from tqdm import tqdm
from os.path import join as jp

path_introspect = './data/vqaintrospect'

basename_introspect = 'VQAIntrospect_<>_withidrelv1.0.json'
basename_introspect_output = 'VQAIntrospect_<>_no_duplicates_withidrelv1.0.json'
subsets = ['train', 'val']

for s in subsets:
    # read json file
    with open(jp(path_introspect, basename_introspect.replace('<>', s)), 'r') as f:
        data = json.load(f)

    # now go through every entry and remove duplicates
    for main_id, entry in tqdm(data.items()):
        all_sub_questions = [] # to save all sub-questions and check against duplicates
        main_question = entry['reasoning_question']
        # list all sub-questions for current entry
        sub_questions = []
        introspect_entries = entry['introspect']
        new_introspect_entries = [] # to save all questions as if they belong to the first annotator
        if len(introspect_entries) == 0:
            continue
        for ie in introspect_entries: # for each annotator
            sub_qa_entries = ie['sub_qa']
            if len(sub_qa_entries) == 0:
                continue
            for sqae in sub_qa_entries: # each sub-question for current annotator
                if sqae['sub_question'] not in all_sub_questions:
                    new_introspect_entries.append({'sub_qa': [sqae], 'pred_q_type': 'reasoning'})
                    all_sub_questions.append(sqae['sub_question'])

        entry['introspect'] = new_introspect_entries

    # save data with output basename
    with open(jp(path_introspect, basename_introspect_output.replace('<>', s)), 'w') as f:
        json.dump(data, f)