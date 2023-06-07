# Project:
#   VQA
# Description:
#   VQA dataset classes
# Author: 
#   Sergio Tascon-Morales

import pickle
import os 
import json
import random
from os.path import join as jp
import torch
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms

from . import vqa2_aux as vqa2_helper
from . import vqa_regions_aux as vqar_helper
from misc import io

class VQABase(Dataset):
    def __init__(self, subset, config, dataset_visual):
        self.subset = subset 
        self.config = config
        self.dataset_visual = dataset_visual
        self.path_annotations_and_questions = jp(config['path_qa'], 'qa')
        self.path_processed = jp(config['path_qa'], 'processed')
        if not os.path.exists(self.path_processed) or len(os.listdir(self.path_processed))<1 or (subset == 'train' and config['process_qa_again']):
            self.pre_process_qa() # pre-process qa, produce pickle files

        # load pre-processed qa
        self.read_prep_rocessed(self.path_processed)

    def pre_process_qa(self):
        raise NotImplementedError # to be implemented in baby class

    def read_prep_rocessed(self, path_files):
        # define paths
        path_map_index_word = jp(path_files, 'map_index_word.pickle')
        path_map_word_index = jp(path_files, 'map_word_index.pickle')
        path_map_index_answer = jp(path_files, 'map_index_answer.pickle')
        path_map_answer_index = jp(path_files, 'map_answer_index.pickle')
        path_dataset = jp(path_files, self.subset + 'set.pickle')

        # read files
        with open(path_map_index_word, 'rb') as f:
                    self.map_index_word = pickle.load(f)
        with open(path_map_word_index, 'rb') as f:
                    self.map_word_index = pickle.load(f)
        with open(path_map_index_answer, 'rb') as f:
                    self.map_index_answer = pickle.load(f)
        with open(path_map_answer_index, 'rb') as f:
                    self.map_answer_index = pickle.load(f)
        with open(path_dataset, 'rb') as f:
                    self.dataset_qa = pickle.load(f)

        # save unknown answer index 
        self.index_unknown_answer = self.map_answer_index['UNK']

    def __getitem__(self, index):
        sample = {}

        # get qa pair
        item_qa = self.dataset_qa[index]

        # get visual
        sample['visual'] = self.dataset_visual.get_by_name(item_qa['image_name'])['visual']

        # get question
        sample['question_id'] = item_qa['question_id']
        sample['question'] = torch.LongTensor(item_qa['question_word_indexes'])

        # get answer
        sample['answer'] = item_qa['answer_index']
        if 'answer_indexes' in item_qa: # trick so that this class can be used with non-vqa2 data
            sample['answers'] = item_qa['answers_indexes']

        return sample

    def __len__(self):
        return len(self.dataset_qa)
        
# Normal VQA dataset classes

class VQA2(VQABase):
    def __init__(self, subset, config, dataset_visual):
        super().__init__(subset, config, dataset_visual)

    def pre_process_qa(self, mainsub=False):
        # first, read original annotations and questions and reformat
        annotations_train = json.load(open(jp(self.path_annotations_and_questions, 'v2_mscoco_train2014_annotations.json'), 'r'))
        annotations_val   = json.load(open(jp(self.path_annotations_and_questions, 'v2_mscoco_val2014_annotations.json'), 'r'))
        questions_train   = json.load(open(jp(self.path_annotations_and_questions, 'v2_OpenEnded_mscoco_train2014_questions.json'), 'r'))
        questions_val     = json.load(open(jp(self.path_annotations_and_questions, 'v2_OpenEnded_mscoco_val2014_questions.json'), 'r'))
        questions_test    = json.load(open(jp(self.path_annotations_and_questions, 'v2_OpenEnded_mscoco_test2015_questions.json'), 'r'))
        questions_testdev = json.load(open(jp(self.path_annotations_and_questions, 'v2_OpenEnded_mscoco_test-dev2015_questions.json'), 'r'))

        # reformat data (transform image id to image name and count occurence of answers for each question)
        data_train = vqa2_helper.reformat_data(questions_train['questions'], 'train', annotations_train['annotations'], mainsub=mainsub)
        data_val = vqa2_helper.reformat_data(questions_val['questions'], 'val', annotations_val['annotations'], mainsub=mainsub)
        data_testdev = vqa2_helper.reformat_data(questions_testdev['questions'], 'testdev')
        data_test = vqa2_helper.reformat_data(questions_test['questions'], 'test')

        # process text
        sets, maps = vqa2_helper.process_qa(self.config, data_train, data_val, data_test, data_testdev)

        # define paths to save pickle files
        if not os.path.exists(self.path_processed):
            os.mkdir(self.path_processed)
        for name, data in sets.items():
            io.save_pickle(data, jp(self.path_processed, name + '.pickle'))
        for name, data in maps.items():
            io.save_pickle(data, jp(self.path_processed, name + '.pickle'))

class GQAMainSub(VQABase):
    def __init__(self, subset, config, dataset_visual):
        super().__init__(subset, config, dataset_visual)
    
    def pre_process_qa(self):
        # method to process qa pairs and save processed files
        # read csv
        self.pairs_df = pd.read_csv(jp(self.path_annotations_and_questions, 'qa.csv'))

        # encode answers and questions so that getitem can provide encoded answers and questions.
        sets, maps = vqa2_helper.process_qa_gqa(self.config, self.pairs_df)

        if not os.path.exists(self.path_processed):
            os.mkdir(self.path_processed)
        for name, data in sets.items():
            io.save_pickle(data, jp(self.path_processed, name + '.pickle'))
        for name, data in maps.items():
            io.save_pickle(data, jp(self.path_processed, name + '.pickle'))

    def __getitem__(self, index):
        samples = [] # a list of dictionaries
        item = self.dataset_qa[index]
        samples = [{    'visual': self.dataset_visual.get_by_name(str(item['img_id']) + '.jpg')['visual'],
                        'question_id': item['mq_id'],
                        'question': torch.LongTensor(item['mq_word_indexes']),
                        'answer': item['ma_index'],
                        #'answers': item['ma_index'],
                        'flag': 1}, 
                        {'visual': self.dataset_visual.get_by_name(str(item['img_id']) + '.jpg')['visual'],
                        'question_id': item['sq_id'],
                        'question': torch.LongTensor(item['sq_word_indexes']),
                        'answer': item['sa_index'],
                        #'answers': item['sa_index'],
                        'flag': 1}]

        return tuple(samples)


    def __len__(self):
        return len(self.dataset_qa)

class VQA2MainSub(VQA2):
    """Same as VQA2 but with main-sub pairs in batches

    """
    def __init__(self, subset, config, dataset_visual, rels=False):
        super().__init__(subset, config, dataset_visual)

        if 'augment' in config:
            self.augment = config['augment']
        else:
            self.augment = False

        self.rels = rels
        self.map_rel_index = {'---': 2, '-->': 1, '<--': 0, 'unk': 2}
        # here, build pairs of main-and sub-questions using self.dataset_qa
        self.dataset_qa_pairs = []
        # list all main questions and all corresponding images
        main_questions = [e for e in self.dataset_qa if e['role']=='main']
        if len(main_questions) < 1:
            raise Exception('For some reason no main questions were detected!')
        print('Building pairs')
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.dataset_qa
        }
        all_subquestions = [e for e in self.dataset_qa if e['role']=='sub']
        assert len(all_subquestions)>0
        self.dataset_qa_pairs = [(self.id2datum[e['parent']], e) for e in all_subquestions]
        """
        for mq in tqdm(main_questions): # iterate through all main questions 
            main_image = mq['image_name']
            # get all sub-questions for current main question
            sub_questions = [e for e in self.dataset_qa if e['image_name'] == main_image and e['role'] == 'sub' and e['parent']==mq['question_id']]
            assert len(sub_questions) >= 1 # sanity check
            for sq in sub_questions:
                self.dataset_qa_pairs.append((mq, sq))
        """        
        # now, append pairs of ind questions
        ind_questions = [e for e in self.dataset_qa if e['role'] == 'ind']
        if len(ind_questions)%2 != 0: # if number of ind pairs is not odd, just append last qa pair again
            ind_questions.append(ind_questions[-1])
        self.dataset_qa_pairs += [(ind_questions[i], ind_questions[i+1]) for i in range(0,len(ind_questions)-1, 2)]

    # override just to set mainsub to true
    def pre_process_qa(self):
        return super().pre_process_qa(mainsub=True)

    # override getitem method
    def __getitem__(self, index):
        samples = [] # a list of dictionaries

        if self.dataset_qa_pairs[index][1]['role'] == 'sub': # get flag and relationship before processing the pair
            good = True # the pair is useful for consistency
            reli = self.map_rel_index[self.dataset_qa_pairs[index][1]['rel']]
        else:
            good = False

        # get qa pair
        for item in self.dataset_qa_pairs[index]:
            sample = {}
            # get visual and mask
            sample['visual'] = self.dataset_visual.get_by_name(item['image_name'])['visual']

            if self.augment:
                raise NotImplementedError

            # get questions
            sample['question_id'] = item['question_id']
            sample['question'] = torch.LongTensor(item['question_word_indexes'])

            # get answers
            sample['answer'] = item['answer_index']
            sample['answers'] = item['answers_indexes']

            if self.rels:
                if good:
                    sample['rel'] = reli
                else:
                    sample['rel'] = self.map_rel_index['---']

            if item['role'] == 'ind':
                sample['flag'] = 0
            else:
                sample['flag'] = 1
                
            samples.append(sample)

        return tuple(samples) # this will return a list of two samples. Collater function in dataloader will have to handle this to build a batch        

    def __len__(self):
        return len(self.dataset_qa_pairs)
        

# Regions VQA dataset classes

def draw_borders(img, coords, r=2):
    # set values to minimum value first
    img[:, coords[2]-r:coords[0]+r, coords[3]-r: coords[3]+r] = torch.min(img)
    img[:, coords[2]-r:coords[0]+r, coords[1]-r: coords[1]+r] = torch.min(img)
    img[:, coords[2]-r: coords[2]+r, coords[3]-r:coords[1]+r] = torch.min(img)
    img[:, coords[0]-r: coords[0]+r, coords[3]-r:coords[1]+r] = torch.min(img)

    # set red channel line to red
    img[0, coords[2]-r:coords[0]+r, coords[3]-r: coords[3]+r] = torch.max(img)
    img[0, coords[2]-r:coords[0]+r, coords[1]-r: coords[1]+r] = torch.max(img)
    img[0, coords[2]-r: coords[2]+r, coords[3]-r:coords[1]+r] = torch.max(img)
    img[0, coords[0]-r: coords[0]+r, coords[3]-r:coords[1]+r] = torch.max(img)

    return img

class VQARegionsSingle(VQABase):
    """Class for dataloader that contains questions about a single region

    Parameters
    ----------
    VQABase : Parent class
        Base class for VQA dataset.
    """
    def __init__(self, subset, config, dataset_visual, dataset_mask, draw_borders=False, rels=False):
        super().__init__(subset, config, dataset_visual)
        self.dataset_mask = dataset_mask
        if 'augment' not in config:
            config['augment'] = False
        self.augment = config['augment']
        self.draw_borders = draw_borders


    def transform(self, image, mask, size):

        if self.subset == 'train': # for training images, do random crops and horizontal flips. For other sets, only resize
            # Random crop
            #i, j, h, w = transforms.RandomCrop.get_params(
            #    image, output_size=(size, size))
            #image = TF.crop(image, i, j, h, w)
            #mask = TF.crop(mask, i, j, h, w)

            # Random horizontal flipping
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            if random.random() > 0.5:
                angle = random.randint(-10, 10) # define angle
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)

        # Transform to tensor
        if not torch.is_tensor(image):
            image = TF.to_tensor(image)
        if not torch.is_tensor(mask):
            mask = TF.to_tensor(mask)
        return image, mask

    # override getitem method
    def __getitem__(self, index):
        sample = {}

        # get qa pair
        item_qa = self.dataset_qa[index]

        # get visual
        visual = self.dataset_visual.get_by_name(item_qa['image_name'])['visual']
        mask = self.dataset_mask.get_by_name(item_qa['mask_name'])['mask']
        if self.augment:
            sample['visual'], sample['maskA'] = self.transform(visual, mask, 448)
        else:
            sample['visual'] = visual
            sample['maskA'] = mask

        # ! commenting for use in DME dataset        sample['coords'] = item_qa['mask_coords']

        #! same             if self.draw_borders:
        #!                       sample['visual'] = draw_borders(sample['visual'], sample['coords'])

        # get question
        sample['question_id'] = item_qa['question_id']
        sample['question'] = torch.LongTensor(item_qa['question_word_indexes'])

        # get answer
        sample['answer'] = item_qa['answer_index']

        return sample

    # define preprocessing method for qa pairs
    def pre_process_qa(self):
        data_train = json.load(open(jp(self.path_annotations_and_questions, 'trainqa.json'), 'r'))
        data_val = json.load(open(jp(self.path_annotations_and_questions, 'valqa.json'), 'r'))
        data_test = json.load(open(jp(self.path_annotations_and_questions, 'testqa.json'), 'r'))

        sets, maps = vqa2_helper.process_qa(self.config, data_train, data_val, data_test)

        # define paths to save pickle files
        if not os.path.exists(self.path_processed):
            os.mkdir(self.path_processed)
        for name, data in sets.items():
            io.save_pickle(data, jp(self.path_processed, name + '.pickle'))
        for name, data in maps.items():
            io.save_pickle(data, jp(self.path_processed, name + '.pickle'))


class VQA2RegionsSingle(VQARegionsSingle):
    def __init__(self, subset, config, dataset_visual, dataset_mask, draw_borders=False):
        super().__init__(subset, config, dataset_visual, dataset_mask, draw_borders=draw_borders)

    # override function so that json qa pairs can be processed differently (taking masks into account)
    def pre_process_qa(self):
        # first, read original annotations and questions and reformat
        annotations_train = json.load(open(jp(self.path_annotations_and_questions, 'v2_mscoco_train2014_annotations.json'), 'r'))
        annotations_val   = json.load(open(jp(self.path_annotations_and_questions, 'v2_mscoco_val2014_annotations.json'), 'r'))
        questions_train   = json.load(open(jp(self.path_annotations_and_questions, 'v2_OpenEnded_mscoco_train2014_questions.json'), 'r'))
        questions_val     = json.load(open(jp(self.path_annotations_and_questions, 'v2_OpenEnded_mscoco_val2014_questions.json'), 'r'))
        questions_test    = json.load(open(jp(self.path_annotations_and_questions, 'v2_OpenEnded_mscoco_test2015_questions.json'), 'r'))
        questions_testdev = json.load(open(jp(self.path_annotations_and_questions, 'v2_OpenEnded_mscoco_test-dev2015_questions.json'), 'r'))

        # reformat data (transform image id to image name and count occurence of answers for each question)
        data_train = vqa2_helper.reformat_data_mask(questions_train['questions'], 'train', annotations_train['annotations'])
        data_val = vqa2_helper.reformat_data_mask(questions_val['questions'], 'val', annotations_val['annotations'])
        data_testdev = vqa2_helper.reformat_data(questions_testdev['questions'], 'testdev')
        data_test = vqa2_helper.reformat_data(questions_test['questions'], 'test')

        # process text
        sets, maps = vqa2_helper.process_qa(self.config, data_train, data_val, data_test, data_testdev)

        # define paths to save pickle files
        if not os.path.exists(self.path_processed):
            os.mkdir(self.path_processed)
        for name, data in sets.items():
            io.save_pickle(data, jp(self.path_processed, name + '.pickle'))
        for name, data in maps.items():
            io.save_pickle(data, jp(self.path_processed, name + '.pickle'))


class VQARegionsSingleMainSub(VQARegionsSingle):
    """Class for getting samples that contain a main question and a subquestion. Dataset must have been prepared in advance 
    using modify_dme_main_questions_sub_questions.py
    """
    def __init__(self, subset, config, dataset_visual, dataset_mask, draw_borders=False, rels=False):
        super().__init__(subset, config, dataset_visual, dataset_mask, draw_borders, rels)

        # here, build pairs of main-and sub-questions using self.dataset_qa
        self.dataset_qa_pairs = []
        self.rels = rels
        self.map_rel_index = {'---': 2, '-->': 1, '<--': 0, '<->': 3}
        # list all main questions and all corresponding images
        main_questions = [e for e in self.dataset_qa if e['role']=='main']
        print('Building pairs')
        for mq in tqdm(main_questions): # iterate through all main questions
            main_image = mq['image_name']
            # get all sub-questions for current main question
            sub_questions = [e for e in self.dataset_qa if e['image_name'] == main_image and e['role'] == 'sub']
            for sq in sub_questions:
                self.dataset_qa_pairs.append((mq, sq))
        # now, append pairs of ind questions
        ind_questions = [e for e in self.dataset_qa if e['role'] == 'ind']
        if len(ind_questions)%2 != 0: # if number of ind pairs is not odd, just append last qa pair again
            ind_questions.append(ind_questions[-1])
        self.dataset_qa_pairs += [(ind_questions[i], ind_questions[i+1]) for i in range(0,len(ind_questions)-1, 2)]


    # override getitem method
    def __getitem__(self, index):
        samples = [] # a list of dictionaries

        if self.dataset_qa_pairs[index][1]['role'] == 'sub': # get flag and relationship before processing the pair
            good = True # the pair is useful for consistency
            reli = self.map_rel_index[self.dataset_qa_pairs[index][1]['rel']]
        else:
            good = False

        # get qa pair
        for item in self.dataset_qa_pairs[index]:
            sample = {}
            # get visual and mask
            visual = self.dataset_visual.get_by_name(item['image_name'])['visual']
            mask = self.dataset_mask.get_by_name(item['mask_name'])['mask']


            if self.augment:
                sample['visual'], sample['maskA'] = self.transform(visual, mask, 448)
            else:
                sample['visual'] = visual
                sample['maskA'] = mask

            # get questions
            sample['question_id'] = item['question_id']
            sample['question'] = torch.LongTensor(item['question_word_indexes'])

            # get answers
            sample['answer'] = item['answer_index']

            if self.rels:
                if good:
                    sample['rel'] = reli
                else:
                    sample['rel'] = self.map_rel_index['---'] # treat any unuseful pair as unrelated

            if item['role'] == 'ind':
                sample['flag'] = 0
            else:
                sample['flag'] = 1
                
            samples.append(sample)

        return tuple(samples) # this will return a list of two samples. Collater function in dataloader will have to handle this to build a batch

    def __len__(self):
        return len(self.dataset_qa_pairs)


class VQARegionsComplementary(VQABase):

    def __init__(self, subset, config, dataset_visual, dataset_maskA, dataset_maskB):
        super().__init__(subset, config, dataset_visual)
        self.dataset_maskA = dataset_maskA
        self.dataset_maskB = dataset_maskB

    # override getitem method
    def __getitem__(self, index):
        sample = {}

        # get qa pair
        item_qa = self.dataset_qa[index]

        # get visual
        sample['visual'] = self.dataset_visual.get_by_name(item_qa['image_name'])['visual']
        sample['maskA'] = self.dataset_maskA.get_by_name(item_qa['maskA_name'])['mask']
        sample['maskB'] = self.dataset_maskB.get_by_name(item_qa['maskB_name'])['mask']

        # get question
        sample['question_id'] = item_qa['question_id']
        sample['question'] = torch.LongTensor(item_qa['question_word_indexes'])

        # get answer
        sample['answer'] = item_qa['answer_index']
        sample['coords'] = torch.LongTensor(item_qa['mask_coords'])

        return sample

    # define preprocessing method for qa pairs
    def pre_process_qa(self):
        data_train = json.load(open(jp(self.path_annotations_and_questions, 'trainqa.json'), 'r'))
        data_val = json.load(open(jp(self.path_annotations_and_questions, 'valqa.json'), 'r'))
        data_test = json.load(open(jp(self.path_annotations_and_questions, 'testqa.json'), 'r'))

        sets, maps = vqa2_helper.process_qa(self.config, data_train, data_val, data_test)

        # define paths to save pickle files
        if not os.path.exists(self.path_processed):
            os.mkdir(self.path_processed)
        for name, data in sets.items():
            io.save_pickle(data, jp(self.path_processed, name + '.pickle'))
        for name, data in maps.items():
            io.save_pickle(data, jp(self.path_processed, name + '.pickle'))