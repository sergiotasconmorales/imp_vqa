
import json
from tqdm import tqdm
from os.path import join as jp

path_rels_annotated  = '../data/introspectnli_2.0' # sentence pairs that were annotated manually
basename_rels_annnotated = '{}_sentence_pairs_relationships_withid.json'
path_rels_predicted = '../data/sentences_dataset'
basename_rels_predicted = '{}_predicted_relationships.json'
path_introspect = '../data/vqaintrospect'
basename_introspect = 'VQAIntrospect_{}_withidv1.0.json'

subsets = ['train', 'val']

def homogenize(rels):
    # function to homogenize dict fields in relationships. Reason: annotated and predicted relationships have different names for the relationship key
    # and also some relationships are denoted as <- instead of <--
    for e in rels:
        # change key name to relationship
        if 'predicted' in e:
            e['relationship'] = e.pop('predicted')
    # now correct notation
    for e in rels:
        if e['relationship'] == '<-':
            e['relationship'] = '<--'
        elif e['relationship'] == '->':
            e['relationship'] = '-->'
        elif e['relationship'] == '-x-':
            e['relationship'] = '---' #! Assigning contradictions as unrelated for simplicity (only 12 cases)
        else:
            pass
    return

# for each subset do
for s in subsets:
    # read introspect file
    with open(jp(path_introspect, basename_introspect.format(s)), 'r') as f:
        intr = json.load(f)
    # read annotated relationships
    with open(jp(path_rels_annotated, basename_rels_annnotated.format(s)), 'r') as f:
        rels_annotated = json.load(f)['samples']
    # read predicted relationships
    with open(jp(path_rels_predicted, basename_rels_predicted.format(s)), 'r') as f:
        rels_predicted = json.load(f)

    # join annotated and predicted relationships
    rels = rels_annotated + rels_predicted
    homogenize(rels)

    # build dict with (main_id, sub_id): rel
    map_ids_rel = {(e['sentence1id'], e['sentence2id']): e['relationship'] for e in rels}

    # now go through all introspect entries and add relationship to sub-question
    new_introspect = {}
    for question_id, contents in tqdm(intr.items()):
        introspect_entries = contents['introspect']
        if len(introspect_entries) == 0:
            continue
        for ie in introspect_entries:
            sub_qa_entries = ie['sub_qa']
            if len(sub_qa_entries) == 0:
                continue
            for sqae in sub_qa_entries:
                ids = (int(question_id), sqae['subquestion_id'])
                if ids in map_ids_rel:
                    curr_rel = map_ids_rel[ids]
                    sqae['rel'] = curr_rel
                    new_introspect[question_id] = contents # includes id and rel
                else:
                    # non-binary QA paris won't be in the map dict. For these, give a special token
                    sqae['rel'] = 'unk'

    # save new version of introspect
    with open(jp(path_introspect, basename_introspect.format(s).replace('withid', 'withidrel')), 'w') as f:
        json.dump(intr, f)