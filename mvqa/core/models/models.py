# Project:
#   VQA
# Description:
#   Model classes definition
# Author: 
#   Sergio Tascon-Morales

import torch
import torch.nn as nn
from torchvision import transforms
from .components.attention import apply_attention
from .components import image, text, attention, fusion, classification


class VQA_Base(nn.Module):
    # base class for simple VQA model
    def __init__(self, config, vocab_words, vocab_answers):
        super().__init__()
        self.visual_feature_size = config['visual_feature_size']
        self.question_feature_size = config['question_feature_size']
        self.pre_visual = config['pre_extracted_visual_feat']
        self.use_attention = config['attention']
        self.number_of_glimpses = config['number_of_glimpses']
        self.visual_size_before_fusion = self.visual_feature_size # 2048 by default, changes if attention
        # Create modules for the model

        # if necesary, create module for offline visual feature extraction
        if not self.pre_visual:
            self.image = image.get_visual_feature_extractor(config)

        # create module for text feature extraction
        self.text = text.get_text_feature_extractor(config, vocab_words)

        # if necessary, create attention module
        if self.use_attention:
            self.visual_size_before_fusion = self.number_of_glimpses*self.visual_feature_size
            self.attention_mechanism = attention.get_attention_mechanism(config)
        else:
            self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))

        # create multimodal fusion module
        self.fuser, fused_size = fusion.get_fuser(config['fusion'], self.visual_size_before_fusion, self.question_feature_size)

        # create classifier
        self.classifer = classification.get_classfier(fused_size, config)


    def forward(self, v, q):
        # if required, extract visual features from visual input 
        if not self.pre_visual:
            v = self.image(v) # [B, 2048, 14, 14]

        # l2 norm
        v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8)
        
        # extract text features
        q = self.text(q)

        # if required, apply attention
        if self.use_attention:
            v = self.attention_mechanism(v, q) # should apply attention too
        else:
            v = self.avgpool(v).squeeze_() # [B, 2048]

        # apply multimodal fusion
        fused = self.fuser(v, q)

        # apply MLP
        x = self.classifer(fused)

        return x
        

class Classifier(nn.Module):

    def __init__(self, config, num_classes):
        super().__init__()
        self.visual_feature_size = config['visual_feature_size']
        self.hidden_size = config['classifier_hidden_size']
        dropout_percentage = config['classifier_dropout']

    # create elements
        self.image = image.get_visual_feature_extractor(config)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.classifer = classification.Classifier(self.visual_feature_size, self.hidden_size, num_classes, drop=dropout_percentage)

    def forward(self, v, m):

        v = self.image(torch.mul(v, m)) # [B, 2048, 14, 14]   MASK IS INCLUDED HERE

        v = self.avgpool(v).squeeze_() # [B, 2048]
        
        x = self.classifer(v)

        return x

class VQARS_1(VQA_Base):
    # First model for region-based VQA, with single mask. Input image is multiplied with the mask to produced a masked version which is sent to the model as normal
    def __init__(self, config, vocab_words, vocab_answers):
        # call mom
        super().__init__(config, vocab_words, vocab_answers)

    # override forward method to accept mask
    def forward(self, v, q, m):
        # if required, extract visual features from visual input 
        if self.pre_visual:
            raise ValueError("This model does not allow pre-extracted features")
        else:
            v = self.image(torch.mul(v, m)) # [B, 2048, 14, 14]   MASK IS INCLUDED HERE

        # extract text features
        q = self.text(q)

        # if required, apply attention
        if self.use_attention:
            v = self.attention_mechanism(v, q) # should apply attention too
        else:
            v = self.avgpool(v).squeeze_(dim=-1).squeeze_(dim=-1) # [B, 2048]

        # apply multimodal fusion
        fused = self.fuser(v, q)

        # apply MLP
        x = self.classifer(fused)

        return x




class SQuINT(VQARS_1):
    # SQuINTed version of model 1
    def __init__(self, config, vocab_words, vocab_answers):
        super().__init__(config, vocab_words, vocab_answers)

    # override forward so that attention maps are returned too
    def forward(self, v, q, m):
        # if required, extract visual features from visual input 
        if self.pre_visual:
            raise ValueError("This model does not allow pre-extracted features")
        else:
            v = self.image(torch.mul(v, m)) # [B, 2048, 14, 14]   MASK IS INCLUDED HERE

        # extract text features
        q = self.text(q)

        # if required, apply attention
        if self.use_attention:
            v, maps = self.attention_mechanism(v, q, return_maps=True) # should apply attention too
        else:
            raise ValueError("Attention is necessary for SQuINT")

        # apply multimodal fusion
        fused = self.fuser(v, q)

        # apply MLP
        x = self.classifer(fused)

        return x, maps


class SQuINT_1(VQA_Base):
    # SQuINTed version of model 7 for current problem. Idea is that reasoning question (grading) should focus on the same areas as the perception questions (all other questions).
    # Model should return the answers for both questions as well as the attention maps.
    def __init__(self, config, vocab_words, vocab_answers):
        if not config['attention']:
            raise ValueError("This model requires attention. Please set <attention> to True in the config file")

        # call mom
        super().__init__(config, vocab_words, vocab_answers)

        # replace attention mechanism
        self.attention_mechanism = attention.get_attention_mechanism(config, special='Att3')

    # override forward method to accept two questions and two masks
    def forward(self, v, q_m, q_s, m_m, m_s):
        # if required, extract visual features from visual input 
        if self.pre_visual:
            raise ValueError("This model does not allow pre-extracted features")
        else:
            visual_features = self.image(v) 

        # FIRST PASS FOR MAIN QUESTION

        # extract text features
        q_m = self.text(q_m)

        # resize mask
        m_m = transforms.Resize(14)(m_m) #should become size (B,1,14,14)
        m_m = m_m.view(m_m.size(0),-1, 14*14) # [B,1,196]

        # if required, apply attention
        if self.use_attention:
            v_m, att_m = self.attention_mechanism(visual_features, m_m, q_m, return_maps=True) 
        else:
            raise ValueError("This model requires attention")

        # apply multimodal fusion
        fused = self.fuser(v_m, q_m)

        # apply MLP
        x_m = self.classifer(fused)


        # SECOND PASS FOR MAIN QUESTION

        # extract text features
        q_s = self.text(q_s)

        # resize mask
        m_s = transforms.Resize(14)(m_s) #should become size (B,1,14,14)
        m_s = m_s.view(m_s.size(0),-1, 14*14) # [B,1,196]

        # if required, apply attention
        if self.use_attention:
            v_s, att_s = self.attention_mechanism(visual_features, m_s, q_s, return_maps=True) 
        else:
            raise ValueError("This model requires attention")

        # apply multimodal fusion
        fused = self.fuser(v_s, q_s)

        # apply MLP
        x_s = self.classifer(fused)

        return x_m, x_s, att_m, att_s



class SQuINT_2(VQA_Base):
    # SQuINTed version of model 7 for current problem. Since diagram and explanation in Selvaraju et al. are so confusing, I have to try things. In this case
    # I will use the attention maps of the sub-question to answer both the main and the sub question, hoping that this is closer to what they did. 

    def __init__(self, config, vocab_words, vocab_answers):
        if not config['attention']:
            raise ValueError("This model requires attention. Please set <attention> to True in the config file")

        # call mom
        super().__init__(config, vocab_words, vocab_answers)

        # replace attention mechanism
        self.attention_mechanism = attention.get_attention_mechanism(config, special='Att3')

    # override forward method to accept two questions and two masks
    def forward(self, v, q_m, q_s, m_m, m_s):
        # if required, extract visual features from visual input 
        if self.pre_visual:
            raise ValueError("This model does not allow pre-extracted features")
        else:
            visual_features = self.image(v) 

        # FIRST PASS FOR MAIN QUESTION

        # extract text features for both main and sub question
        q_m = self.text(q_m)
        q_s = self.text(q_s)

        # resize masks
        m_m = transforms.Resize(14)(m_m) #should become size (B,1,14,14)
        m_m = m_m.view(m_m.size(0),-1, 14*14) # [B,1,196]
        m_s = transforms.Resize(14)(m_s) #should become size (B,1,14,14)
        m_s = m_s.view(m_s.size(0),-1, 14*14) # [B,1,196]

        # if required, apply attention
        if self.use_attention:
            _, att_m = self.attention_mechanism(visual_features, m_m, q_m, return_maps=True) 
            v_s, att_s = self.attention_mechanism(visual_features, m_s, q_s, return_maps=True) 
        else:
            raise ValueError("This model requires attention")

        # apply multimodal fusion to attended visual features from sub question and main question embedding (as done in Selvaraju et al.)
        fused = self.fuser(v_s, q_m)

        # apply MLP
        x_m = self.classifer(fused)

        # SECOND PASS FOR MAIN QUESTION

        # apply multimodal fusion to attended visual features from sub question and sub question embedding (as done in Selvaraju et al.)
        fused = self.fuser(v_s, q_s)

        # apply MLP
        x_s = self.classifer(fused)

        return x_m, x_s, att_m, att_s


class SQuINT_3(VQA_Base):
    # Trying to get a model that produces good accuracy by processing two questions at a time. 

    def __init__(self, config, vocab_words, vocab_answers):
        if not config['attention']:
            raise ValueError("This model requires attention. Please set <attention> to True in the config file")

        # call mom
        super().__init__(config, vocab_words, vocab_answers)

        # replace attention mechanism
        self.attention_mechanism = attention.get_attention_mechanism(config, special='Att3')

    # override forward method to accept two questions and two masks
    def forward(self, v, q_m, q_s, m_m, m_s):
        # if required, extract visual features from visual input 
        if self.pre_visual:
            raise ValueError("This model does not allow pre-extracted features")
        else:
            visual_features = self.image(v) 

        # FIRST PASS FOR MAIN QUESTION

        # extract text features for both main and sub question
        q_m = self.text(q_m)
        q_s = self.text(q_s)

        # resize masks
        m_m = transforms.Resize(14)(m_m) #should become size (B,1,14,14)
        m_m = m_m.view(m_m.size(0),-1, 14*14) # [B,1,196]
        m_s = transforms.Resize(14)(m_s) #should become size (B,1,14,14)
        m_s = m_s.view(m_s.size(0),-1, 14*14) # [B,1,196]

        # if required, apply attention
        if self.use_attention:
            _, att_m = self.attention_mechanism(visual_features, m_m, q_m, return_maps=True) 
            v_s, att_s = self.attention_mechanism(visual_features, m_s, q_s, return_maps=True) 
        else:
            raise ValueError("This model requires attention")

        # apply multimodal fusion to attended visual features from sub question and main question embedding (as done in Selvaraju et al.)
        fused = self.fuser(v_s, q_m)

        # apply MLP
        x_m = self.classifer(fused)

        # SECOND PASS FOR MAIN QUESTION

        # apply multimodal fusion to attended visual features from sub question and sub question embedding (as done in Selvaraju et al.)
        fused = self.fuser(v_s, q_s)

        # apply MLP
        x_s = self.classifer(fused)

        return x_m, x_s, att_m, att_s

