

import json
import os
from os.path import join as jp
import stanza
import nltk
import copy
from tqdm import tqdm
from nltk.tokenize import word_tokenize
import pandas as pd
from nltk.tokenize.treebank import TreebankWordDetokenizer


def is_binary(answer):
    if answer in ['yes', 'no', 'yeah', 'nope', 'yea']:
        return True
    else:
        return False

def correct_answer_format(answer):
    if answer in ['yeah', 'yea', 'yep']:
        return 'yes'
    elif answer in ['nope', 'nop']:
        return 'no'
    else:
        return answer

def create_dir(*args):
    for arg in args:
        if not os.path.exists(arg):
            os.mkdir(arg)
        else:
            print('Directory', arg, 'exists already')

def switch_answer(answer):
    if answer == 'yes':
        return 'no'
    elif answer == 'no':
        return 'yes'
    else:
        raise ValueError

def swap(li, pos1, pos2):
    li[pos1], li[pos2] = li[pos2], li[pos1]
    return li

def swap_and_insert(tokens, answer):
    swap(tokens, 0, 1)
    if answer == 'no':
        tokens.insert(2, 'not')
    return

def split_word(word):
    return [char for char in word]


class SentencesDataset(object):
    def __init__(self, path_introspect, path_output, base_filename = 'VQAIntrospect_<>v1.0.json', pos_tag_engine = 'stanza', path_verbs = 'verbs/most-common-verbs-english.csv'):
        self.path_introspect = path_introspect
        self.path_output = path_output
        create_dir(self.path_output)
        self.path_aux = jp(path_output, 'aux')
        create_dir(self.path_aux)
        self.base_filename = base_filename
        verbs = pd.read_csv(path_verbs)
        self.dict_verbs = {verbs.iloc[i]['Word']: verbs.iloc[i]['3singular'] for i in range(verbs.shape[0])}

        self.subsets = ['train', 'val']
        self.first_words = ['is', 'are', 'does', 'do', 'can'] # first words for which a conversion into sentences will be implemented
        nltk.download('averaged_perceptron_tagger')
        self.pos_tag_engine = pos_tag_engine
        if pos_tag_engine == 'stanza':
            self.nlp = stanza.Pipeline('en')

    def add_question_ids(self):
        # add questions ids to the subquestions in introspect. In VQA2 the maximum length for an id is 9 digits, so I will create ids as int(str(cnt) + '0'*9)
        cnt = 1 # counter for ids
        for s in self.subsets:
            with open(jp(self.path_introspect, 'VQAIntrospect_<>v1.0.json'.replace('<>', s)), 'r') as f:
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
            with open(jp(self.path_introspect, self.base_filename.replace('<>', s)), 'w') as f:
                json.dump(data, f)

    def _is_valid_question(self, question):
        # determines if a question is valid by checking if it starts with a word that is present in self.first_words
        if question.split(' ')[0] in self.first_words:
            return True
        else:
            return False

    def _get_dict(self, image_id, mq, ma, mid, sq, sa, sid):
        # builds dictionary with path to image, main question, main answer, sub-question and sub answer
        info = {'image_id': image_id,
                'main_question': mq,
                'main_answer': ma,
                'main_id': mid,
                'sub_question': sq,
                'sub_answer': sa,
                'sub_id': sid}
        return info

    def _save_binary_introspect(self, subset):
        # first, if file was created already, return
        if os.path.exists(jp(self.path_aux, subset + '.json')):
            return

        # else, open VQA-Introspect data and generate the corresponding file with QA pairs that are valid (binary)
        with open(jp(self.path_introspect, self.base_filename.replace('<>', subset)), 'r') as f:
            data = json.load(f)
        entries = []
        for question_id, contents in data.items():
            main_q = contents['reasoning_question']
            main_a = contents['reasoning_answer_most_common']
            if not is_binary(main_a): # if main question is not binary, then continue
                continue
            image_id = contents['image_id']
            introspect_entries = contents['introspect']
            if len(introspect_entries) == 0:
                continue
            for ie in introspect_entries:
                sub_qa_entries = ie['sub_qa']
                if len(sub_qa_entries) == 0:
                    continue
                for sqae in sub_qa_entries:
                    answer = sqae['sub_answer']
                    subq_id = sqae['subquestion_id']
                    if answer == 'yes' or answer == 'yea' or answer == 'yeah' or answer == 'no' or answer == 'nope':
                        entries.append(self._get_dict(image_id, main_q, correct_answer_format(main_a), int(question_id), sqae['sub_question'], correct_answer_format(sqae['sub_answer']), subq_id)) 
                    else:
                        pass

        # save binary entries to aux path
        with open(jp(self.path_aux, subset + '.json'), 'w') as f:
            json.dump(entries, f)

    def _read_binary_introspect(self, subset):
        # read json file from path_aux
        with open(jp(self.path_aux, subset + '.json'), 'r') as f:
            self.entries = json.load(f)

    def _find_children(self, id_s, ids, heads, chil):
        # finds direct children of id_s
        if id_s in heads:
            idx_children = [i for i,x in enumerate(heads) if x == id_s]
            children_ids = [ids[i] for i in idx_children]
        else:
            children_ids = []
        return children_ids


    def _tokenize_pos_tag(self, question):
        if self.pos_tag_engine == 'nltk':
            tokens =  word_tokenize(question)
            pos_tags = nltk.pos_tag(tokens)
        elif self.pos_tag_engine == 'stanza':
            if '.?' in question: # stanza is dumb when question ends with e.g. u.s.? (does not separate them) so in this case remove the question mark
                question = question.replace('?', '')
            doc = self.nlp(question)
            if '?' in question:
                tokens = [e.text for s in doc.sentences for e in s.tokens[:-1]]
                tags = [e.xpos for s in doc.sentences for e in s.words[:-1]]
                # dependency parsing
                deprels = [e.deprel for s in doc.sentences for e in s.words[:-1]]
                ids = [e.id for s in doc.sentences for e in s.words[:-1]]
                heads = [e.head for s in doc.sentences for e in s.words[:-1]]
            else:
                tokens = [e.text for s in doc.sentences for e in s.tokens]
                tags = [e.xpos for s in doc.sentences for e in s.words]
                # dependency parsing
                deprels = [e.deprel for s in doc.sentences for e in s.words]
                ids = [e.id for s in doc.sentences for e in s.words]
                heads = [e.head for s in doc.sentences for e in s.words]

            pos_tags = [(tok, tag) for tok, tag in zip(tokens, tags)]
            dict_id_head = {i:h for i,h in zip(ids, heads)}
            num_offspring = 0
            if 'nsubj' in deprels:
                # if there is subj, then count the len of the set tha contains it along with it's offspring
                index_nsubj = deprels.index('nsubj')
                id_nsubj = ids[index_nsubj]
                # now find ids of offspring
                num_offspring = 1
                # iterate through all ids
                ids_explore = copy.copy(ids)
                ids_explore.remove(id_nsubj)
                for curr_id in ids_explore:
                    destiny = curr_id
                    cnt=0
                    while destiny != 0 and destiny != id_nsubj:
                        if curr_id in dict_id_head: # some sentences have big typos
                            destiny = dict_id_head[curr_id]
                            curr_id = destiny
                        if cnt == 100: # max iterations, prevent infinite loop
                            break
                        cnt+=1
                    if destiny == id_nsubj:
                        num_offspring += 1

        return tokens, pos_tags, num_offspring

    def _amount_n_after(self, pos_tags):
        cnt = 0
        for e in pos_tags:
            if split_word(e[1])[0] == 'N':
                cnt+=1
        return cnt

    def get_index_end_object(self, pos_tags, tokens, answer):
        found = False
        once = False
        for i in range(1, len(tokens)-1):
            if pos_tags[i][1] == 'PRP': # if a personal pronoun is found, that should be the object the questions refers to
                found = True
                break
            elif split_word(pos_tags[i][1])[0] == 'N': # found a noun, but it might not be the whole noun (e.g. ice cream, water bowl). 
                if pos_tags[i+1][1] in ['IN', 'CC'] and self._amount_n_after(pos_tags[i+1:]) == 1: 
                    break # if there is only one N after the IN/CC, then take current N as N for sentence
                elif pos_tags[i+1][1] in ['IN', 'CC'] and self._amount_n_after(pos_tags[i+1:]) > 1:
                    once = True
                    continue # if found first Noun followed by IN or CC, just skip it
                for j in range(i+1, len(tokens)): # keep looking for N from next token
                    if split_word(pos_tags[j][1])[0] != 'N' and pos_tags[j][1] != 'POS': # if no longer N, this is the index (unless it's a Possessive ending)
                        i = j-1
                        found = True
                        break
                    if j == len(tokens)-1: # last word is also a NN (e.g. is the ice cream vanilla?) but "are these chocolate donuts?" has a problem because <are> is inserted after chocolate.
                        i = j-1
                        found = True
                        break
                if found:
                    break
        first_token = tokens.pop(0) # remove first token
        if len(tokens) == 1: 
            i = 0 # if there is only one token (e.g. evening), i is probably not assigned, so put the first token at the beginning (e.g. is evening)
        tokens.insert(i, first_token) # insert first token (is/are) after object  
        if answer == 'no':
            tokens.insert(i+1, 'not')
        return

    def _verb_present(self, pos_tags):
        # checks if at least one of the tokens (different from the first one) is a verb
        verb_present = False
        for i in range(1, len(pos_tags)):
            if split_word(pos_tags[i][1])[0] == 'V':
                verb_present = True
        return verb_present

    def _sentensize(self, question, answer):
        # remove question mark

        
        # get tokens pos tags
        tokens, pos_tagged_text, offspring = self._tokenize_pos_tag(question)
        

        # now build sentences depending on first word of the question
        if tokens[0] in ['is', 'are']:
            # check if tokens[1] is 'there' 
            if tokens[1] == 'there':
                swap_and_insert(tokens, answer)
            else:
                # here I have to use the pos tags to get the whole object to which "is" refers. Then I have to send "is" after the object's name.
                # consider different objects such as "this" and "the red car of the right"
                if tokens[1] in ['this', 'that', 'these', 'those']:
                    if split_word(pos_tagged_text[2][1])[0] != 'N' or pos_tagged_text[2][1] == 'PRP': # if word after this/that is not a noun, it means the this/that does not refer to a specific object in the sentence
                        swap_and_insert(tokens, answer)
                    else: # questions like "is this car red?"
                        #special case: is this _? 
                        if len(tokens) == 3:
                            swap_and_insert(tokens, answer)
                        elif split_word(pos_tagged_text[2][1])[0] == 'N': # word after this/that/these/those is noun, so assume is a one-word noun (TODO: should look for the end e.g. is that boy in red a football player?)
                            first_token = tokens.pop(0)
                            tokens.insert(2, first_token)
                            if answer=='no':
                                tokens.insert(3, 'not')
                        else:
                            if offspring > 0:
                                first_token = tokens.pop(0)
                                tokens.insert(offspring, first_token)
                                if answer=='no':
                                    tokens.insert(offspring+1, 'not')
                            else: # no nsubj
                                self.get_index_end_object(pos_tagged_text, tokens, answer)                   
                else:
                    if offspring > 0:
                        first_token = tokens.pop(0)
                        tokens.insert(offspring, first_token)
                        if answer=='no':
                            tokens.insert(offspring+1, 'not')
                    else:
                        # here, consider all words betwen position 1 and x to be the object. So send the "is" to a position after the first tag containing N
                        self.get_index_end_object(pos_tagged_text, tokens, answer)  

        elif tokens[0] in ['does', 'do']:
            # if answer is yes, simply remove the does/do
            if answer == 'yes':
                first_token = tokens.pop(0)
                if first_token == 'does':
                    for i in range(len(pos_tagged_text)):
                        if pos_tagged_text[i][1] == 'VB':
                            if pos_tagged_text[i][0] in self.dict_verbs:
                                tokens[i-1] = self.dict_verbs[pos_tagged_text[i][0]] # minus one because one was popped
                                break
                        
            # if answer is no, build structure
            else: 
                for i in range(1, len(tokens)-1):
                    if split_word(pos_tagged_text[i][1])[0] == 'V' and split_word(pos_tagged_text[i+1][1])[0] != 'V': # look for verb that comes after object
                        index_noun = i
                        break
                    if i==len(tokens)-2: # for cases like do the bananas talk? verb is at end, so move one position to avoid errors like the do not bananas talk
                        i += 1
                        break
                first_token = tokens.pop(0) # remove first token
                tokens.insert(i-1, first_token) # insert first token (is/are) after object  
                tokens.insert(i, 'not')
        elif tokens[0] == 'can':
            for i in range(1, len(tokens)):
                if split_word(pos_tagged_text[i][1])[0] == 'V': # search for verb (different from can)
                    index_noun = i
                    break
            tokens.pop(0) # remove can
            tokens.insert(i-1, 'can')

            if answer == 'no':
                tokens[i-1] = 'cannot'

        else:
            return

        sentence = TreebankWordDetokenizer().detokenize(tokens)
        return sentence # TODO return the first word too

    def _fill_output(self, output_dict, mq, ma, sq, sa):
        # first, consistent sentences
        output_dict['sentence1'] = self._sentensize(mq, ma)
        output_dict['sentence2'] = self._sentensize(sq, sa)

        return output_dict      

    def _create_sentences(self, entry):
        mq = entry['main_question']
        ma = entry['main_answer']
        sq = entry['sub_question']
        sa = entry['sub_answer']
        mid = entry['main_id']
        sid = entry['sub_id']

        output = {'sentence1': None, 'sentence1id': mid, 'sentence2': None, 'sentence2id': sid, 'label': 1, 'image_id': entry['image_id']} 
        output = self._fill_output(output, mq, ma, sq, sa)

        return output

    def _convert_pairs_into_sentences(self, subset, balance):
        # consider only question starting with is/are, do/does or can. Maybe there will be correlation between first word of sub-question and consistency label
        sentence_pairs = []
        for entry in tqdm(self.entries):
            # i have to consider only those cases in which both main and sub can be converted
            m_valid = self._is_valid_question(entry['main_question'])
            s_valid = self._is_valid_question(entry['sub_question'])
            if m_valid and s_valid: # if both main and sub-question can be converted into sentences
                # proceed
                if balance:
                    # here I need to list all possible sub-questions for each "first word". Then omit some based on the number of sub-questions of the least represented answer
                    raise NotImplementedError
                else:
                    pair = self._create_sentences(entry)
                    sentence_pairs.append(pair) 

        # save sentence pairs
        with open(jp(self.path_output, subset + '_sentence_pairs.json'), 'w') as f:
            json.dump(sentence_pairs, f)

    def create_dataset(self, balance = False):
        for s in self.subsets:
            # first, create file with all binary QA pairs from introspect (skip if they exist). Save it
            self._save_binary_introspect(s)

            # read saved files
            self._read_binary_introspect(s)

            # convert every QA pair into sentences
            self._convert_pairs_into_sentences(s, balance)