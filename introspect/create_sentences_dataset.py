import sys
sys.path.append('../')

from bert.dataset.dataset import SentencesDataset

path_introspect = '../data/vqaintrospect'
path_output = '../data/sentences_dataset'

sentences_dataset = SentencesDataset(path_introspect, path_output, base_filename = 'VQAIntrospect_<>_withidv1.0.json', pos_tag_engine = 'stanza')
# first, add ids to introspect dataset
sentences_dataset.add_question_ids()
# now create dataset of propositions
sentences_dataset.create_dataset(balance = False)