# Project:
#   VQA
# Description:
#   Train, validation and test loops
# Author: 
#   Sergio Tascon-Morales

import torch
from torch import nn
from torch.nn.functional import one_hot
from torch._C import Value
from . import train_utils
from metrics import metrics
from .criterions import Q2_score
import time
import numpy as np

REMAPPING = torch.tensor([0,1,2,3,4]).cuda(), torch.tensor([1,0,3,2,4]).cuda()
EPSILON = 1e-10

def train(train_loader, model, criterion, optimizer, device, epoch, config, logbook, comet_exp=None, consistency_term=None):
    
    # set train mode
    model.train()

    # Initialize variables for collecting metrics from all batches
    loss_epoch = 0.0
    running_loss = 0.0
    acc_epoch = 0.0

    for i, sample in enumerate(train_loader):
        batch_size = sample['question'].size(0)

        # move data to GPU
        question = sample['question'].to(device)
        visual = sample['visual'].to(device)
        answer = sample['answer'].to(device)
        answers = sample['answers'].to(device)

        # clear parameter gradients
        optimizer.zero_grad()

        # get output from model
        output = model(visual, question)
        train_utils.sync_if_parallel(config) #* necessary?

        # compute loss
        loss = criterion(output, answer)

        running_loss += loss.item()
        if comet_exp is not None and i%10 == 9: # log every 10 iterations
            comet_exp.log_metric('loss_train_step', running_loss/10, step=len(train_loader)*(epoch-1) + i+1)
            running_loss = 0.0

        # compute accuracy
        acc = metrics.vqa_accuracy(output, answers)

        loss.backward()
        train_utils.sync_if_parallel(config) #* necessary?
        optimizer.step()
        train_utils.sync_if_parallel(config) #* necessary?

        # laters: save to logger and print 
        loss_epoch += loss.item()
        acc_epoch += acc.item()

    metrics_dict = {'loss_train': loss_epoch/len(train_loader.dataset), 'acc_train': acc_epoch/len(train_loader.dataset)}

    logbook.log_metrics('train', metrics_dict, epoch)

    return metrics_dict# returning average for all samples


def train_dme(train_loader, model, criterion, optimizer, device, epoch, config, logbook, comet_exp=None, consistency_term=None, rels=None):
    
    # set train mode
    model.train()

    # Initialize variables for collecting metrics from all batches
    loss_epoch = 0.0
    running_loss = 0.0
    acc_epoch = 0.0

    for i, sample in enumerate(train_loader):

        # move data to GPU
        question = sample['question'].to(device)
        visual = sample['visual'].to(device)
        answer = sample['answer'].to(device)
        if 'maskA' in sample:
            mask_a = sample['maskA'].to(device)

        # clear parameter gradients
        optimizer.zero_grad()

        # get output from model
        if 'maskA' in sample:
            output = model(visual, question, mask_a)
        else:
            output = model(visual, question)
        train_utils.sync_if_parallel(config) #* necessary?

        # compute loss
        loss = criterion(output, answer)

        loss.backward()
        train_utils.sync_if_parallel(config) #* necessary?
        optimizer.step()
        train_utils.sync_if_parallel(config) #* necessary?

        running_loss += loss.item()
        if comet_exp is not None and i%10 == 9: # log every 10 iterations
            comet_exp.log_metric('loss_train_step', running_loss/10, step=len(train_loader)*(epoch-1) + i+1)
            running_loss = 0.0

        # compute accuracy
        acc = metrics.batch_strict_accuracy(output, answer)

        # laters: save to logger and print 
        loss_epoch += loss.item()
        acc_epoch += acc.item()

    metrics_dict = {'loss_train': loss_epoch/len(train_loader.dataset), 'acc_train': acc_epoch/len(train_loader.dataset)} 

    logbook.log_metrics('train', metrics_dict, epoch)

    return metrics_dict# returning average for all samples

def compute_mse(maps, flag, crit, device):
    # Function to compute MSE between attention maps (SQuINT)
    if torch.count_nonzero(flag) == 0: # if all flags are 0 (all questions are ind), return 0, otherwise NAN is generated.
        return torch.tensor(0)

    # separate into even and odd indexes (main vs sub or ind vs ind)
    main_maps = torch.index_select(maps, 0, torch.tensor([i for i in range(0, len(maps), 2)]).to(device))
    sub_maps = torch.index_select(maps, 0, torch.tensor([i for i in range(1, len(maps), 2)]).to(device))
    flag_reduced = torch.index_select(flag, 0, torch.tensor([i for i in range(0, len(flag), 2)]).to(device))
    mse = crit(main_maps, sub_maps)
    return torch.mean(mse[torch.where(flag_reduced>0)])

def train_dme_mainsub_squint(train_loader, model, criteria, optimizer, device, epoch, config, logbook, comet_exp=None, consistency_term=None):
    
    # In this case criteria contains two cross entropies: one as usual for right answers and a second one without reduction
    criterion1, criterion2 = criteria

    # set train mode
    model.train()

    # Initialize variables for collecting metrics from all batches
    loss_epoch = 0.0
    running_loss = 0.0
    acc_epoch = 0.0
    mse_accum = 0.0

    for i, sample in enumerate(train_loader):

        # move data to GPU
        question = sample['question'].to(device)
        visual = sample['visual'].to(device)
        answer = sample['answer'].to(device)
        mask_a = sample['maskA'].to(device)
        flag = sample['flag'].to(device)

        # clear parameter gradients
        optimizer.zero_grad()

        # get output from model
        output, att_maps = model(visual, question, mask_a)
        train_utils.sync_if_parallel(config) #* necessary?

        # build term for inconsistency reduction
        mse = compute_mse(att_maps, flag, criterion2, device) 
        mse_accum += mse.item()

        loss = criterion1(output, answer) + config['lambda']*mse

        loss.backward()
        train_utils.sync_if_parallel(config) #* necessary?
        optimizer.step()
        train_utils.sync_if_parallel(config) #* necessary?

        running_loss += loss.item()
        if comet_exp is not None and i%10 == 9: # log every 10 iterations
            comet_exp.log_metric('loss_train_step', running_loss/10, step=len(train_loader)*(epoch-1) + i+1)
            running_loss = 0.0

        # compute accuracy
        acc = metrics.batch_strict_accuracy(output, answer)

        # laters: save to logger and print 
        loss_epoch += loss.item()
        acc_epoch += acc.item()

    if config['mainsub']:
        denominator_acc = 2*len(train_loader.dataset)
    else:
        denominator_acc = len(train_loader.dataset)

    metrics_dict = {'loss_train': loss_epoch/len(train_loader), 'acc_train': acc_epoch/denominator_acc, 'mse_train': mse_accum/len(train_loader)}

    logbook.log_metrics('train', metrics_dict, epoch)

    return metrics_dict# returning average for all samples
def rels_loss_wrapper(ce, flag_reduced):
    if flag_reduced.sum() > 0:
        return torch.mean(torch.index_select(ce, 0, torch.nonzero(flag_reduced).squeeze(-1)))
    else:
        return 0


def train_dme_mainsub_rels(train_loader, model, criteria, optimizer, device, epoch, config, logbook, comet_exp=None, consistency_term=None, rels=False):
    """Attempt to reduce inconsistencies in a different way than that proposed in Selvaraju et al"""

    #* In this case criteria contains two cross entropies: one as usual for right answers and a second one without reduction
    criterion1, criterion2 = criteria

    # set train mode
    model.train()

    # Initialize variables for collecting metrics from all batches
    loss_epoch = 0.0
    running_loss = 0.0
    acc_epoch = 0.0
    q2 = 0.0

    # if from second epoch, define beta for whole epoch
    if consistency_term.adaptive and epoch > 1:
        consistency_term.update_loss_params()

    for i, sample in enumerate(train_loader):

        # move data to GPU
        question = sample['question'].to(device) # [B, 23]
        visual = sample['visual'].to(device) # [B, 3, 448, 448]
        answer = sample['answer'].to(device) # [B]
        if 'maskA' in sample:
            mask_a = sample['maskA'].to(device)
        flag = sample['flag'].to(device) # [B]
        flag_reduced = flag[::2] # reduced version of flag
        if 'rel' in sample:
            relations = sample['rel'].to(device)

        # clear parameter gradients
        optimizer.zero_grad()

        # get output from model
        if 'maskA' in sample:
            output, rels_pred = model(visual, question, mask_a)
        else:
            output = model(visual, question)
        train_utils.sync_if_parallel(config) #* necessary?

        loss = config['alpha']*criterion1(output, answer) + config['beta']*rels_loss_wrapper(criterion2(rels_pred, relations_ind), flag_reduced)
        

        loss.backward()
        train_utils.sync_if_parallel(config) #* necessary?
        optimizer.step()
        train_utils.sync_if_parallel(config) #* necessary?

        running_loss += loss.item()
        if comet_exp is not None and i%10 == 9: # log every 10 iterations
            comet_exp.log_metric('loss_train_step', running_loss/10, step=len(train_loader)*(epoch-1) + i+1)
            running_loss = 0.0

        # compute accuracy
        acc = metrics.batch_strict_accuracy(output, answer)

        # laters: save to logger and print 
        loss_epoch += loss.item()
        acc_epoch += acc.item()

    if config['mainsub']:
        denominator_acc = 2*len(train_loader.dataset)
    else:
        denominator_acc = len(train_loader.dataset)

    metrics_dict = {'loss_train': loss_epoch/len(train_loader), 'acc_train': acc_epoch/denominator_acc}

    logbook.log_metrics('train', metrics_dict, epoch)

    return metrics_dict# returning average for all samples

def remap_values(remapping, x):
    index = torch.bucketize(x.ravel(), remapping[0])
    return remapping[1][index].reshape(x.shape)

def get_inconsistent_indexes(rels, answers):
    # function to compute the indexes of the answers that would produce inconsistent pairs (i.e. negated answers corresponding to necessary conditions)
    answers_copy = torch.clone(answers)
    main_ans = answers_copy[::2]
    sub_ans = answers_copy[1::2]
    rels_reduced = rels[::2]
    main_ans_toggled = remap_values(REMAPPING, main_ans)
    sub_ans_toggled = remap_values(REMAPPING, sub_ans)
    if 0 in rels_reduced:
        main_ans[rels_reduced == 0] = main_ans_toggled[rels_reduced==0]
    if 1 in rels_reduced:
        sub_ans[rels_reduced == 1] = sub_ans_toggled[rels_reduced==1]
    return answers_copy

def inconsistency_loss1(probs, flag):
    main_probs = probs[::2]
    sub_probs = probs[1::2]
    flag_reduced = flag[::2]
    func = torch.log(1-main_probs + EPSILON)*torch.log(1-sub_probs + EPSILON)
    return torch.mean(func[torch.where(flag_reduced>0)])

def inconsistency_loss2(probs, flag, sigma=0.9):
    main_probs = probs[::2]
    sub_probs = probs[1::2]
    flag_reduced = flag[::2]
    func = torch.exp(-((main_probs - 1)**2 + (sub_probs - 1)**2)/(2*sigma**2)) #*(1/(2*torch.pi*sigma**2))
    return torch.mean(func[torch.where(flag_reduced>0)])

def inconsistency_loss3(probs, flag):
    main_probs = probs[::2]
    sub_probs = probs[1::2]
    flag_reduced = flag[::2]
    func = -main_probs*torch.log(1-sub_probs + EPSILON) -sub_probs*torch.log(1-main_probs + EPSILON)
    return torch.mean(func[torch.where(flag_reduced>0)])


# more general approach: incorrect prob is the complement of the probability of the answer in the example
softmax = nn.Softmax(dim=1)
def consistency_loss(scores, target, rel, config):

    prob = softmax(scores)
    assert prob.shape[0] == target.shape[0] == rel.shape[0]
    if torch.sum(rel) == 2*rel.shape[0]: # no useful pairs
        return torch.tensor(0)
    # get main and sub parts of everything
    prob_main = prob[::2, :]
    prob_sub = prob[1::2, :]
    target_main = target[::2]
    target_sub = target[1::2]
    rel_red = rel[::2]

    p = torch.zeros_like(rel_red, dtype=torch.float32).cuda()
    q = torch.zeros_like(rel_red, dtype=torch.float32).cuda()
    p_ = torch.zeros_like(rel_red, dtype=torch.float32).cuda()
    q_ = torch.zeros_like(rel_red, dtype=torch.float32).cuda()

    # For <-- relations (0 index in rel)
    p[rel_red==0] = 1 - (prob_main[rel_red==0]*one_hot(target_main[rel_red==0], num_classes = prob_main.shape[1]).to(torch.float32)).sum(1)
    q[rel_red==0] = (prob_sub[rel_red==0]*one_hot(target_sub[rel_red==0], num_classes = prob_sub.shape[1]).to(torch.float32)).sum(1)

    # For --> relations (1 index in rel)
    p[rel_red==1] = (prob_main[rel_red==1]*one_hot(target_main[rel_red==1], num_classes = prob_main.shape[1]).to(torch.float32)).sum(1)
    q[rel_red==1] = 1 - (prob_sub[rel_red==1]*one_hot(target_sub[rel_red==1], num_classes = prob_sub.shape[1]).to(torch.float32)).sum(1)

    # For <-> relations (3 index in rel)
    p[rel_red==3] = 1 - (prob_main[rel_red==3]*one_hot(target_main[rel_red==3], num_classes = prob_main.shape[1]).to(torch.float32)).sum(1)
    q[rel_red==3] = (prob_sub[rel_red==3]*one_hot(target_sub[rel_red==3], num_classes = prob_sub.shape[1]).to(torch.float32)).sum(1)
    p_[rel_red==3] = (prob_main[rel_red==3]*one_hot(target_main[rel_red==3], num_classes = prob_main.shape[1]).to(torch.float32)).sum(1)
    q_[rel_red==3] = 1 - (prob_sub[rel_red==3]*one_hot(target_sub[rel_red==3], num_classes = prob_sub.shape[1]).to(torch.float32)).sum(1)

    flag_valid = (rel_red!=2).to(torch.int64)

    if config['inc_loss'] == 'inconsistency_loss1':
        return torch.mean((torch.log(1-p + EPSILON)*torch.log(1-q + EPSILON))[torch.where(flag_valid>0)]) + torch.mean((torch.log(1-p_ + EPSILON)*torch.log(1-q_ + EPSILON))[torch.where(flag_valid>0)])
    elif config['inc_loss'] == 'inconsistency_loss2':
        sigma = 0.4
        return torch.mean(torch.exp(-((p - 1)**2 + (q - 1)**2)/(2*sigma**2))[torch.where(flag_valid>0)]) + torch.mean(torch.exp(-((p_ - 1)**2 + (q_ - 1)**2)/(2*sigma**2))[torch.where(flag_valid>0)])
    elif config['inc_loss'] == 'inconsistency_loss3':
        func = -p*torch.log(1-q + EPSILON) -q*torch.log(1-p + EPSILON) + -p_*torch.log(1-q_ + EPSILON) -q_*torch.log(1-p_ + EPSILON)
        return torch.mean(func[torch.where(flag_valid>0)])

def train_dme_mainsub_consistrain(train_loader, model, criteria, optimizer, device, epoch, config, logbook, comet_exp=None, consistency_term=None, rels=False):
    """Attempt to reduce inconsistencies in a different way than that proposed in Selvaraju et al"""

    #* In this case criteria contains two cross entropies: one as usual for right answers and a second one without reduction
    criterion1, criterion2 = criteria

    # set train mode
    model.train()

    # Initialize variables for collecting metrics from all batches
    loss_epoch = 0.0
    running_loss = 0.0
    acc_epoch = 0.0
    q2 = 0.0

    # if from second epoch, define beta for whole epoch
    if consistency_term.adaptive and epoch > 1:
        consistency_term.update_loss_params()

    softm = nn.Softmax(dim=1)

    for i, sample in enumerate(train_loader):

        # move data to GPU
        question = sample['question'].to(device) # [B, 23]
        visual = sample['visual'].to(device) # [B, 3, 448, 448]
        answer = sample['answer'].to(device) # [B]
        if 'maskA' in sample:
            mask_a = sample['maskA'].to(device)
        flag = sample['flag'].to(device) # [B]
        if 'rel' in sample:
            relations = sample['rel'].to(device)

        # clear parameter gradients
        optimizer.zero_grad()

        # get output from model
        if 'maskA' in sample:
            output = model(visual, question, mask_a)
        else:
            output = model(visual, question)
        train_utils.sync_if_parallel(config) #* necessary?

        # build term for inconsistency reduction
        non_avg_ce = criterion2(output, answer)
        
        # depending on function of consistency term, proceed as required
        if consistency_term.adaptive:
            # update max CE for main questions so that beta can be updated properly for next epoch
            consistency_term.log_ces_sub_main(non_avg_ce, flag, device)

            if epoch == 1: # in first epoch, do not include consistency term because you don't have a good estimate for beta
                loss = criterion1(output, answer)
            else:
                if config['consistency_function'] == 'fcn10':
                    q2_incons = config['lambda']*consistency_term.compute_loss_term_nlp(question, output, flag, device) # mq, ma, sq, sa
                else: 
                    q2_incons = config['lambda']*consistency_term.compute_loss_term(non_avg_ce, flag, device)
                loss = criterion1(output, answer) + q2_incons
                q2 += q2_incons.item()
        else:
            if config['consistency_function'] == 'fcn10':
                q2_incons = config['lambda']*consistency_term.compute_loss_term_nlp(question, output, flag, device) # mq, ma, sq, sa
            else: 
                if rels:
                    if torch.count_nonzero(flag) == 0:
                        q2_incons = torch.tensor(0)
                    else:

                        q2_incons = config['lambda']*consistency_loss(output, answer, relations, config)
                        if comet_exp is not None:
                            comet_exp.log_metric('q2_iter_train', q2_incons.item(), step=len(train_loader)*(epoch-1) + i+1)
                else:
                    q2_incons = config['lambda']*consistency_term.compute_loss_term(non_avg_ce, flag, device)
            loss = criterion1(output, answer) + q2_incons
            q2 += q2_incons.item()
            
        loss.backward()
        train_utils.sync_if_parallel(config) #* necessary?
        optimizer.step()
        train_utils.sync_if_parallel(config) #* necessary?

        running_loss += loss.item()
        if comet_exp is not None and i%10 == 9: # log every 10 iterations
            comet_exp.log_metric('loss_train_step', running_loss/10, step=len(train_loader)*(epoch-1) + i+1)
            running_loss = 0.0

        # compute accuracy
        acc = metrics.batch_strict_accuracy(output, answer)

        # laters: save to logger and print 
        loss_epoch += loss.item()
        acc_epoch += acc.item()

    if config['mainsub']:
        denominator_acc = 2*len(train_loader.dataset)
    else:
        denominator_acc = len(train_loader.dataset)

    metrics_dict = {'loss_train': loss_epoch/len(train_loader), 'acc_train': acc_epoch/denominator_acc, 'q2_train': q2/len(train_loader)}

    logbook.log_metrics('train', metrics_dict, epoch)

    return metrics_dict# returning average for all samples




def validate(val_loader, model, criterion, device, epoch, config, logbook, comet_exp=None, consistency_term=None):
    # tensor to save results
    results = torch.zeros((len(val_loader.dataset), 2), dtype=torch.int64)

    # set evaluation mode
    model.eval()

    # Initialize variables for collecting metrics from all batches
    loss_epoch = 0.0
    running_loss = 0.0
    acc_epoch = 0.0

    offset = 0
    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            batch_size = sample['question'].size(0)

            # move data to GPU
            question = sample['question'].to(device)
            visual = sample['visual'].to(device)
            answer = sample['answer'].to(device)
            answers = sample['answers'].to(device)
            question_indexes = sample['question_id'] # keep in cpu

            # get output
            output = model(visual, question)

            # compute loss
            loss = criterion(output, answer)

            # compute accuracy
            acc = metrics.vqa_accuracy(output, answers)

            # save answer indexes and answers
            sm = nn.Softmax(dim=1)
            probs = sm(output)
            _, pred = probs.max(dim=1) 
            results[offset:offset+batch_size,0] = question_indexes 
            results[offset:offset+batch_size,1] = pred
            offset += batch_size

            loss_epoch += loss.item()
            acc_epoch += acc.item()

    metrics_dict = {'loss_val': loss_epoch/len(val_loader.dataset), 'acc_val': acc_epoch/len(val_loader.dataset)}
    if logbook is not None:
        logbook.log_metrics('val', metrics_dict, epoch)

    return metrics_dict, results # returning averages for all samples

def validate_dme(val_loader, model, criterion, device, epoch, config, logbook, comet_exp=None, consistency_term=None, rels=None):

    #if config['mainsub']:
    #    denominator_acc = 2*len(val_loader.dataset)
    #else:
    denominator_acc = len(val_loader.dataset)

    # tensor to save results
    results = torch.zeros((denominator_acc, 2), dtype=torch.int64)

    # set evaluation mode
    model.eval()

    # Initialize variables for collecting metrics from all batches
    loss_epoch = 0.0
    acc_epoch = 0.0

    offset = 0
    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            batch_size = sample['question'].size(0)

            # move data to GPU
            question = sample['question'].to(device)
            visual = sample['visual'].to(device)
            answer = sample['answer'].to(device)
            question_indexes = sample['question_id'] # keep in cpu
            if 'maskA' in sample:
                mask_a = sample['maskA'].to(device)
                output = model(visual, question, mask_a)
                if isinstance(output, tuple):
                    output = output[0]
            else:
                output  = model(visual, question)
            # get output

            if 'squint' in config: # special case for squint
                if config['squint']:
                    output = output[0]
                

            # compute loss
            if isinstance(criterion, tuple):
                loss = criterion[0](output, answer)
            else:
                loss = criterion(output, answer)

            # compute accuracy
            acc = metrics.batch_strict_accuracy(output, answer)

            # save answer indexes and answers
            sm = nn.Softmax(dim=1)
            probs = sm(output)
            _, pred = probs.max(dim=1)
        
            results[offset:offset+batch_size,0] = question_indexes
            results[offset:offset+batch_size,1] = pred
            offset += batch_size

            loss_epoch += loss.item()
            acc_epoch += acc.item()

    metrics_dict = {'loss_val': loss_epoch/denominator_acc, 'acc_val': acc_epoch/denominator_acc} #! averaging by number of mini-batches
    if logbook is not None:
        logbook.log_metrics('val', metrics_dict, epoch)

    return metrics_dict, results # returning averages for all samples


def validate_dme_mainsub_squint(val_loader, model, criteria, device, epoch, config, logbook, comet_exp=None, consistency_term=None):
 #* In this case criteria contains two cross entropies: one as usual for right answers and a second one without reduction
    criterion1, criterion2 = criteria

    if config['mainsub']:
        denominator_acc = 2*len(val_loader.dataset)
    else:
        denominator_acc = len(val_loader.dataset)

    # tensor to save results
    results = torch.zeros((denominator_acc, 2), dtype=torch.int64)

    # set evaluation mode
    model.eval()

    # Initialize variables for collecting metrics from all batches
    loss_epoch = 0.0
    acc_epoch = 0.0
    mse_accum = 0.0

    offset = 0
    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            batch_size = sample['question'].size(0)

            # move data to GPU
            question = sample['question'].to(device)
            visual = sample['visual'].to(device)
            answer = sample['answer'].to(device)
            question_indexes = sample['question_id'] # keep in cpu
            mask_a = sample['maskA'].to(device)
            flag = sample['flag'].to(device)

            # get output
            output, att_maps = model(visual, question, mask_a)
            train_utils.sync_if_parallel(config) #* necessary?

            # build term for inconsistency reduction
            mse = compute_mse(att_maps, flag, criterion2, device)
            mse_accum += mse.item()

            loss = criterion1(output, answer) + config['lambda']*mse

            # compute accuracy
            acc = metrics.batch_strict_accuracy(output, answer)

            # save answer indexes and answers
            sm = nn.Softmax(dim=1)
            probs = sm(output)
            _, pred = probs.max(dim=1)
        
            results[offset:offset+batch_size,0] = question_indexes
            results[offset:offset+batch_size,1] = pred
            offset += batch_size

            loss_epoch += loss.item()
            acc_epoch += acc.item()

    metrics_dict = {'loss_val': loss_epoch/len(val_loader), 'acc_val': acc_epoch/denominator_acc, 'mse_val': mse_accum/len(val_loader)} 
    if logbook is not None:
        logbook.log_metrics('val', metrics_dict, epoch)

    return metrics_dict, results # returning averages for all samples

def validate_dme_mainsub_rels(val_loader, model, criteria, device, epoch, config, logbook, comet_exp=None, consistency_term=None, rels=False):
    #* In this case criteria contains two cross entropies: one as usual for right answers and a second one without reduction
    criterion1, criterion2 = criteria

    if config['mainsub']:
        denominator_acc = 2*len(val_loader.dataset)
    else:
        denominator_acc = len(val_loader.dataset)

    # tensor to save results
    results = torch.zeros((denominator_acc, 2), dtype=torch.int64)

    # set evaluation mode
    model.eval()

    # Initialize variables for collecting metrics from all batches
    loss_epoch = 0.0
    acc_epoch = 0.0
    q2 = 0.0

    offset = 0
    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            batch_size = sample['question'].size(0)

            # move data to GPU
            question = sample['question'].to(device)
            visual = sample['visual'].to(device)
            answer = sample['answer'].to(device)
            question_indexes = sample['question_id'] # keep in cpu
            if 'maskA' in sample:
                mask_a = sample['maskA'].to(device)
            flag = sample['flag'].to(device)
            flag_reduced = flag[::2] # reduced version of flag
            if 'rel' in sample:
                relations = sample['rel'].to(device)
                reduced_relations = relations[::2, :]
                relations_ind = torch.argmax(reduced_relations, dim=1) + reduced_relations.sum(dim=1)

            # get output
            if 'maskA' in sample:
                output, rels_pred = model(visual, question, mask_a)
            else:
                output = model(visual, question)    

            loss = config['alpha']*criterion1(output, answer) + config['beta']*rels_loss_wrapper(criterion2(rels_pred, relations_ind), flag_reduced)

            # compute accuracy
            acc = metrics.batch_strict_accuracy(output, answer)

            # save answer indexes and answers
            sm = nn.Softmax(dim=1)
            probs = sm(output)
            _, pred = probs.max(dim=1)
        
            results[offset:offset+batch_size,0] = question_indexes
            results[offset:offset+batch_size,1] = pred
            offset += batch_size

            loss_epoch += loss.item()
            acc_epoch += acc.item()

    metrics_dict = {'loss_val': loss_epoch/len(val_loader), 'acc_val': acc_epoch/denominator_acc} 
    if logbook is not None:
        logbook.log_metrics('val', metrics_dict, epoch)

    return metrics_dict, results # returning averages for all samples

def validate_dme_mainsub_consistrain(val_loader, model, criteria, device, epoch, config, logbook, comet_exp=None, consistency_term=None, rels=False):

    #* In this case criteria contains two cross entropies: one as usual for right answers and a second one without reduction
    criterion1, criterion2 = criteria

    if config['mainsub']:
        denominator_acc = 2*len(val_loader.dataset)
    else:
        denominator_acc = len(val_loader.dataset)

    # tensor to save results
    results = torch.zeros((denominator_acc, 2), dtype=torch.int64)

    # set evaluation mode
    model.eval()

    # Initialize variables for collecting metrics from all batches
    loss_epoch = 0.0
    acc_epoch = 0.0
    q2 = 0.0

    softm = nn.Softmax(dim=1)

    offset = 0
    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            batch_size = sample['question'].size(0)

            # move data to GPU
            question = sample['question'].to(device)
            visual = sample['visual'].to(device)
            answer = sample['answer'].to(device)
            question_indexes = sample['question_id'] # keep in cpu
            if 'maskA' in sample:
                mask_a = sample['maskA'].to(device)
            flag = sample['flag'].to(device)

            if 'rel' in sample:
                relations = sample['rel'].to(device)

            # get output
            if 'maskA' in sample:
                output = model(visual, question, mask_a)
            else:
                output = model(visual, question)

            # build term for inconsistency reduction
            non_avg_ce = criterion2(output, answer)

            if consistency_term.adaptive:
                if epoch == 1: # in first epoch, do not include consistency term because you don't have a good estimate for beta
                    loss = criterion1(output, answer)
                else:
                    if config['consistency_function'] == 'fcn10':
                        q2_incons = config['lambda']*consistency_term.compute_loss_term_nlp(question, output, flag, device)
                    else:
                        q2_incons = config['lambda']*consistency_term.compute_loss_term(non_avg_ce, flag, device)
                    loss = criterion1(output, answer) + q2_incons
                    q2 += q2_incons.item()
            else:
                if config['consistency_function'] == 'fcn10':
                    q2_incons = config['lambda']*consistency_term.compute_loss_term_nlp(question, output, flag, device)
                else:
                    if rels:
                        if torch.count_nonzero(flag) == 0:
                            q2_incons = torch.tensor(0)
                        else:
                            q2_incons = config['lambda']*consistency_loss(output, answer, relations, config)
                            if comet_exp is not None:
                                comet_exp.log_metric('q2_iter_val', q2_incons.item(), step=len(val_loader)*(epoch-1) + i+1)
                    else:
                        q2_incons = config['lambda']*consistency_term.compute_loss_term(non_avg_ce, flag, device)
                loss = criterion1(output, answer) + q2_incons
                q2 += q2_incons.item()

            # compute accuracy
            acc = metrics.batch_strict_accuracy(output, answer)

            # save answer indexes and answers
            sm = nn.Softmax(dim=1)
            probs = sm(output)
            _, pred = probs.max(dim=1)
        
            results[offset:offset+batch_size,0] = question_indexes
            results[offset:offset+batch_size,1] = pred
            offset += batch_size

            loss_epoch += loss.item()
            acc_epoch += acc.item()

    metrics_dict = {'loss_val': loss_epoch/len(val_loader), 'acc_val': acc_epoch/denominator_acc, 'q2_val': q2/len(val_loader)} 
    if logbook is not None:
        logbook.log_metrics('val', metrics_dict, epoch)

    return metrics_dict, results # returning averages for all samples


# functions for binary VQA
def train_binary(train_loader, model, criterion, optimizer, device, epoch, config, logbook, comet_exp=None, consistency_term=None):
    # tensor to save results
    results = torch.zeros((len(train_loader.dataset), 2), dtype=torch.int64) # to store question id, model's answer
    answers = torch.zeros(len(train_loader.dataset), 2) # store target answer, prob

    # set train mode
    model.train()

    # Initialize variables for collecting metrics from all batches
    loss_epoch = 0.0
    acc_epoch = 0.0
    offset = 0
    for i, sample in enumerate(train_loader):
        batch_size = sample['question'].size(0)

        # move data to GPU
        question = sample['question'].to(device)
        visual = sample['visual'].to(device)
        answer = sample['answer'].to(device)
        question_indexes = sample['question_id'] # keep in cpu
        if 'maskA' in sample:
            mask_a = sample['maskA'].to(device)
        if 'maskB' in sample:
            mask_b = sample['maskB'].to(device)

        # clear parameter gradients
        optimizer.zero_grad()
        # get output from model
        if 'maskB' in sample:
            output = model(visual, question, mask_a, mask_b)
        else:
            if 'maskA' in sample:
                output = model(visual, question, mask_a)
            else:
                output = model(visual, question)

        train_utils.sync_if_parallel(config) #* necessary?

        # compute loss
        loss = criterion(output.squeeze_(dim=-1), answer.float()) # cast to float because of BCEWithLogitsLoss 

        loss.backward()
        train_utils.sync_if_parallel(config) #* necessary?
        optimizer.step()
        train_utils.sync_if_parallel(config) #* necessary?
        # add running loss
        loss_epoch += loss.item()  
        # save probs and answers
        m = nn.Sigmoid()
        pred = m(output.data.cpu()) 
        # compute accuracy
        acc = metrics.batch_binary_accuracy((pred > 0.5).float().to(device), answer)
        results[offset:offset+batch_size,:] = torch.cat((question_indexes.view(batch_size, 1), torch.round(pred.view(batch_size,1))), dim=1)
        answers[offset:offset+batch_size] = torch.cat((answer.data.cpu().view(batch_size, 1), pred.view(batch_size,1)), dim=1)
        offset += batch_size
        acc_epoch += acc.item()

    # compute AUC and AP for current epoch
    auc, ap = metrics.compute_auc_ap(answers)
    metrics_dict = {'loss_train': loss_epoch/len(train_loader.dataset), 'auc_train': auc, 'ap_train': ap, 'acc_train': acc_epoch/len(train_loader.dataset)}
    logbook.log_metrics('train', metrics_dict, epoch)
    return metrics_dict


def train_binary_mainsub(train_loader, model, criteria, optimizer, device, epoch, config, logbook, comet_exp=None, consistency_term=None):
    # tensor to save results


    if config['mainsub']:
        denominator_acc = 2*len(train_loader.dataset)
    else:
        denominator_acc = len(train_loader.dataset)

    answers = torch.zeros(denominator_acc, 2) # store target answer, prob

    criterion1, criterion2 = criteria

    # set train mode
    model.train()

    # Initialize variables for collecting metrics from all batches
    loss_epoch = 0.0
    running_loss = 0.0
    acc_epoch = 0.0    
    q2 = 0.0
    offset = 0

    for i, sample in enumerate(train_loader):
        batch_size = sample['question'].size(0)

        # move data to GPU
        question = sample['question'].to(device)
        visual = sample['visual'].to(device)
        answer = sample['answer'].to(device)
        question_indexes = sample['question_id'] # keep in cpu
        flag = sample['flag'].to(device)
        if 'maska' in sample:
            mask_a = sample['maskA'].to(device)

        # clear parameter gradients
        optimizer.zero_grad()
        # get output from model
        if 'maskA' in sample:
            output = model(visual, question, mask_a)
        else:
            output = model(visual, question)

        train_utils.sync_if_parallel(config) #* necessary?

        non_avg_ce = criterion2(output.squeeze(dim=-1), answer.float())

        if config['consistency_function'] == 'fcn10':
            q2_incons = config['lambda']*consistency_term.compute_loss_term_nlp(question, output, flag, device) # mq, ma, sq, sa
        else: 
            q2_incons = config['lambda']*consistency_term.compute_loss_term(non_avg_ce, flag, device)

        #y_onehot = torch.FloatTensor(8, 2)
        #y_onehot.zero_()
        #y_onehot.scatter_(1, answer.unsqueeze(-1), 1)

        loss = criterion1(output.squeeze(dim=-1), answer.float()) + q2_incons # cast to float because of BCEWithLogitsLoss 
        q2 += q2_incons.item()

        loss.backward()
        train_utils.sync_if_parallel(config) #* necessary?
        optimizer.step()
        train_utils.sync_if_parallel(config) #* necessary?

        # add running loss
        running_loss += loss.item()
        
        if comet_exp is not None and i%10 == 9: # log every 10 iterations
            comet_exp.log_metric('loss_train_step', running_loss/10, step=len(train_loader)*(epoch-1) + i+1)
            running_loss = 0.0

        # save probs and answers
        m = nn.Sigmoid()
        pred = m(output.data.cpu())
        # compute accuracy
        acc = metrics.batch_binary_accuracy((pred > 0.5).float().to(device), answer)

        answers[offset:offset+batch_size] = torch.cat((answer.data.cpu().view(batch_size, 1), pred.view(batch_size,1)), dim=1)

        offset += batch_size
        loss_epoch += loss.item() 
        acc_epoch += acc.item()


    # compute AUC and AP for current epoch
    auc, ap = metrics.compute_auc_ap(answers)
    metrics_dict = {'loss_train': loss_epoch/len(train_loader.dataset), 'auc_train': auc, 'ap_train': ap, 'acc_train': acc_epoch/denominator_acc}
    logbook.log_metrics('train', metrics_dict, epoch)
    return metrics_dict

def validate_binary(val_loader, model, criterion, device, epoch, config, logbook, comet_exp=None, consistency_term=None):
    # tensor to save results
    results = torch.zeros((len(val_loader.dataset), 2), dtype=torch.int64) # to store question id, model's answer
    answers = torch.zeros(len(val_loader.dataset), 2) # store target answer, prob

    # set evaluation mode
    model.eval()

    # Initialize variables for collecting metrics from all batches
    loss_epoch = 0.0
    acc_epoch = 0.0
    offset = 0
    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            batch_size = sample['question'].size(0)

            # move data to GPU
            question = sample['question'].to(device)
            visual = sample['visual'].to(device)
            answer = sample['answer'].to(device)
            question_indexes = sample['question_id'] # keep in cpu
            if 'maskA' in sample:
                mask_a = sample['maskA'].to(device)
            if 'maskB' in sample:
                mask_b = sample['maskB'].to(device)

            # get output from model
            if 'maskB' in sample:
                output = model(visual, question, mask_a, mask_b)
            else:
                if 'maskA' in sample:
                    output = model(visual, question, mask_a)
                else:
                    output = model(visual, question)

            # compute loss
            loss = criterion(output.squeeze_(dim=-1), answer.float())

            # save probs and answers
            m = nn.Sigmoid()
            pred = m(output.data.cpu())
            # compute accuracy
            acc = metrics.batch_binary_accuracy((pred > 0.5).float().to(device), answer)
            results[offset:offset+batch_size,0] = question_indexes
            results[offset:offset+batch_size,1] = torch.round(pred)
            answers[offset:offset+batch_size] = torch.cat((answer.data.cpu().view(batch_size, 1), pred.view(batch_size,1)), dim=1)
            offset += batch_size

            loss_epoch += loss.item()
            acc_epoch += acc.item()
            
    # compute AUC and AP for current epoch for all samples, using info in results
    auc, ap = metrics.compute_auc_ap(answers)
    metrics_dict = {'loss_val': loss_epoch/len(val_loader.dataset), 'auc_val': auc, 'ap_val': ap, 'acc_val': acc_epoch/len(val_loader.dataset)}
    if logbook is not None:
        logbook.log_metrics('val', metrics_dict, epoch)
    return metrics_dict, {'results': results, 'answers': answers}

def validate_binary_mainsub(val_loader, model, criteria, device, epoch, config, logbook, comet_exp=None, consistency_term=None):
    # tensor to save results


    criterion1, criterion2 = criteria

    if config['mainsub']:
        denominator_acc = 2*len(val_loader.dataset)
    else:
        denominator_acc = len(val_loader.dataset)

    results = torch.zeros((denominator_acc, 2), dtype=torch.int64) # to store question id, model's answer
    answers = torch.zeros(denominator_acc, 2) # store target answer, prob

    # set evaluation mode
    model.eval()

    # Initialize variables for collecting metrics from all batches
    loss_epoch = 0.0
    running_loss = 0.0
    acc_epoch = 0.0    
    q2 = 0.0
    offset = 0
    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            batch_size = sample['question'].size(0)

            # move data to GPU
            question = sample['question'].to(device)
            visual = sample['visual'].to(device)
            answer = sample['answer'].to(device)
            question_indexes = sample['question_id'] # keep in cpu
            flag = sample['flag'].to(device)
            if 'maskA' in sample:
                mask_a = sample['maskA'].to(device)

            # get output from model
            if 'maskA' in sample:
                output = model(visual, question, mask_a)
            else:
                output = model(visual, question)

            # build term for inconsistency reduction
            non_avg_ce = criterion2(output.squeeze_(dim=-1), answer.float())

            # compute loss
            if config['consistency_function'] == 'fcn10':
                q2_incons = config['lambda']*consistency_term.compute_loss_term_nlp(question, output, flag, device)
            else:
                q2_incons = config['lambda']*consistency_term.compute_loss_term(non_avg_ce, flag, device)
            loss = criterion1(output.squeeze_(dim=-1), answer.float()) + q2_incons
            q2 += q2_incons.item()

            # add running loss
            running_loss += loss.item()
            
            if comet_exp is not None and i%10 == 9: # log every 10 iterations
                comet_exp.log_metric('loss_val_step', running_loss/10, step=len(val_loader)*(epoch-1) + i+1)
                running_loss = 0.0

            # save probs and answers
            m = nn.Sigmoid()
            pred = m(output.data.cpu())
            # compute accuracy
            acc = metrics.batch_binary_accuracy((pred > 0.5).float().to(device), answer)
            results[offset:offset+batch_size,0] = question_indexes
            results[offset:offset+batch_size,1] = torch.round(pred)
            answers[offset:offset+batch_size] = torch.cat((answer.data.cpu().view(batch_size, 1), pred.view(batch_size,1)), dim=1)
            offset += batch_size

            loss_epoch += loss.item()
            acc_epoch += acc.item()
            
    # compute AUC and AP for current epoch for all samples, using info in results
    auc, ap = metrics.compute_auc_ap(answers)
    metrics_dict = {'loss_val': loss_epoch/len(val_loader.dataset), 'auc_val': auc, 'ap_val': ap, 'acc_val': acc_epoch/denominator_acc}
    if logbook is not None:
        logbook.log_metrics('val', metrics_dict, epoch)
    return metrics_dict, {'results': results, 'answers': answers}

def train_classif(train_loader, model, criterion, optimizer, device, epoch, config, logbook):
    # tensor to save results
    #results = torch.zeros((len(train_loader.dataset), 2), dtype=torch.int64) # to store question id, model's answer
    answers = torch.zeros(len(train_loader.dataset), 2) # store target answer, prob

    # set train mode
    model.train()

    # Initialize variables for collecting metrics from all batches
    loss_epoch = 0.0
    acc_epoch = 0.0
    offset = 0
    for i, sample in enumerate(train_loader):
        batch_size = sample['index_gt'].size(0)

        # move data to GPU
        visual = sample['visual'].to(device)
        label = sample['label'].to(device)
        index_gt = sample['index_gt'].to(device)
        mask_a = sample['maskA'].to(device)

        # clear parameter gradients
        optimizer.zero_grad()

        # get output from model
        output = model(visual, mask_a)
        train_utils.sync_if_parallel(config) #* necessary?

        # compute loss
        loss = criterion(output.gather(1, index_gt.view(-1,1)).squeeze_(), label.float()) # cast to float because of BCEWithLogitsLoss 

        loss.backward()
        train_utils.sync_if_parallel(config) #* necessary?
        optimizer.step()
        train_utils.sync_if_parallel(config) #* necessary?

        # add running loss
        loss_epoch += loss.item()

        # save probs and answers
        m = nn.Sigmoid()
        pred = m(output.gather(1, index_gt.view(-1,1)).squeeze_().data.cpu())
        # compute accuracy
        acc = metrics.batch_binary_accuracy((pred > 0.5).float().to(device), label)

        answers[offset:offset+batch_size] = torch.cat((label.data.cpu().view(batch_size, 1), pred.view(batch_size,1)), dim=1)
        offset += batch_size
        acc_epoch += acc.item()

    # compute AUC and AP for current epoch
    auc, ap = metrics.compute_auc_ap(answers)
    metrics_dict = {'loss_train': loss_epoch/len(train_loader.dataset), 'auc_train': auc, 'ap_train': ap, 'acc_train': acc_epoch/len(train_loader.dataset)}
    logbook.log_metrics('train', metrics_dict, epoch)
    return metrics_dict


def val_classif(val_loader, model, criterion, device, epoch, config, logbook=None):
    # tensor to save results
    #results = torch.zeros((len(val_loader.dataset), 2), dtype=torch.int64) # to store question id, model's answer
    answers = torch.zeros(len(val_loader.dataset), 2) # store target answer, prob

    # set evaluation mode
    model.eval()

    # Initialize variables for collecting metrics from all batches
    loss_epoch = 0.0
    acc_epoch = 0.0
    offset = 0
    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            batch_size = sample['index_gt'].size(0)

            # move data to GPU
            visual = sample['visual'].to(device)
            label = sample['label'].to(device)
            mask_a = sample['maskA'].to(device)
            index_gt = sample['index_gt'].to(device)

            # get output from model
            output = model(visual, mask_a)

            # compute loss
            loss = criterion(output.gather(1, index_gt.view(-1,1)).squeeze_(), label.float())

            # save probs and answers
            m = nn.Sigmoid()
            pred = m(output.gather(1, index_gt.view(-1,1)).squeeze_().data.cpu())
            # compute accuracy
            acc = metrics.batch_binary_accuracy((pred > 0.5).float().to(device), label)

            answers[offset:offset+batch_size] = torch.cat((label.data.cpu().view(batch_size, 1), pred.view(batch_size,1)), dim=1)
            offset += batch_size

            loss_epoch += loss.item()
            acc_epoch += acc.item()
            
    # compute AUC and AP for current epoch for all samples, using info in results
    auc, ap = metrics.compute_auc_ap(answers)
    metrics_dict = {'loss_val': loss_epoch/len(val_loader.dataset), 'auc_val': auc, 'ap_val': ap, 'acc_val': acc_epoch/len(val_loader.dataset)}
    if logbook is not None:
        logbook.log_metrics('val', metrics_dict, epoch)
    return metrics_dict, {'answers': answers}




# FOR NLP
def train_nlp(model, dataloader, criterion, optimizer, epoch, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0


    for i, data in enumerate(dataloader):
        # send data to gpu
        batch_size = data['main_question'].shape[0]
        mq = data['main_question'].to(device)
        ma = data['main_answer'].to(device)
        sq = data['sub_question'].to(device)
        sa = data['sub_answer'].to(device)
        label = data['label'].to(device)

         # zero the parameter gradients
        optimizer.zero_grad()

        outputs = model(mq,ma,sq,sa)
        loss = criterion(outputs.squeeze_(dim=-1), label.float())
        m = nn.Sigmoid()
        pred = m(outputs)
        predicted_labels = (pred>0.5).float()
        acc = metrics.batch_binary_accuracy(label, predicted_labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += acc.item()
    return {'loss_train': running_loss/len(dataloader), 'acc_train': 100*running_acc/len(dataloader.dataset)}

def validate_nlp(model, dataloader, epoch, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    with torch.no_grad():
      for i, data in enumerate(dataloader):
        mq = data['main_question'].to(device)
        ma = data['main_answer'].to(device)
        sq = data['sub_question'].to(device)
        sa = data['sub_answer'].to(device)
        label = data['label'].to(device)
        outputs = model(mq,ma,sq,sa)
          
        loss = criterion(outputs.squeeze_(dim=-1), label.float())
        m = nn.Sigmoid()
        pred = m(outputs)
        predicted_labels = (pred>0.5).float()
        acc = metrics.batch_binary_accuracy(label, predicted_labels)

        running_loss += loss.item()
        running_acc += acc.item()
    return {'loss_val': running_loss/len(dataloader), 'acc_val': 100*running_acc/len(dataloader.dataset)}



def get_looper_functions(config, classif=False, test=False):
    if test:
        return validate_dme, validate_dme
    if classif:
        return train_classif, val_classif

    if config['num_answers'] == 2: # binary 
        if config['mainsub']:
            train_fn = train_binary_mainsub
            val_fn = validate_binary_mainsub
        else:  
            train_fn = train_binary
            val_fn = validate_binary
    else: # non-binary
        if config['mainsub']:
            if 'squint' in config:
                if config['squint']:
                    train_fn = train_dme_mainsub_squint
                    val_fn = validate_dme_mainsub_squint
                else:
                    train_fn = train_dme_mainsub_consistrain
                    val_fn = validate_dme_mainsub_consistrain     
            elif 'relpred' in config:
                if config['relpred']:
                    train_fn = train_dme_mainsub_rels
                    val_fn = validate_dme_mainsub_rels              
            else:
                train_fn = train_dme_mainsub_consistrain
                val_fn = validate_dme_mainsub_consistrain
        elif config['dataset'] == 'vqa2':
            train_fn = train 
            val_fn = validate
        else:
            train_fn = train_dme 
            val_fn = validate_dme

    return train_fn, val_fn
