import os
import sys
from operator import itemgetter

import sklearn
import sklearn.metrics
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from eval import show_results
import scipy
import scipy.stats

import csv

def logit_score(logit, top_k):
    score_list = []
    for idx in range(len(logit)):
        logit_i = logit[idx].data.cpu().numpy().tolist()
        indices, L_sorted = zip(*sorted(enumerate(logit_i), key=itemgetter(1), reverse=True))
        score_i = (top_k * L_sorted[0] - sum(L_sorted[1:top_k + 1])) / top_k
        score_list.append(score_i)
    return score_list

def drop_freq(y_probs):
    y_probs = np.array(y_probs)
    n = y_probs.shape[0]
    drop_score_list = []
    for i in range(y_probs.shape[1]):
        logit_i = y_probs[:, i]
        counts = np.bincount(logit_i)
        max_count = np.max(counts)
        drop_score_list.append(max_count / n)
    return drop_score_list

def drop_entropy(y_prbos, mask_num):
    y_probs = np.array(y_prbos)
    entropy_list = []
    for i in range(y_probs.shape[1]):
        logit_i = y_probs[:, i]
        bin_count = np.bincount(logit_i)
        mask = sorted(range(len(bin_count)), key=lambda i: bin_count[i])[:-mask_num]
        bin_count[mask] = 0
        count_probs = [i / sum(bin_count) for i in bin_count]

        entropy = scipy.stats.entropy(count_probs)
        entropy_list.append(entropy)
    return entropy_list

##below is added by 
def drop_entropy_emnlp(y_probs, mask_num, logit_sfmean, logit_sfvar, represent_setvar, y_probs_2, logit_sfmean_2, logit_sfvar_2, represent_setvar_2):
    y_probs = np.array(y_probs)
    y_probs_2 = np.array(y_probs_2)
    entropy_list = []
    y_inner_confidence = np.max(logit_sfmean, axis = 1).tolist()
    y_inner_confidence_2 = np.max(logit_sfmean_2, axis=1).tolist()
    for i in range(y_probs.shape[1]):
        logit_i = y_probs[:, i]
        bin_count = np.bincount(logit_i)
        mask = sorted(range(len(bin_count)), key=lambda i: bin_count[i])[:-mask_num]
        bin_count[mask] = 0
        count_probs = [i / sum(bin_count) for i in bin_count]

        logit_i_2 = y_probs_2[:, i]
        bin_count_2 = np.bincount(logit_i_2)
        mask_2 = sorted(range(len(bin_count_2)), key=lambda i: bin_count_2[i])[:-mask_num]
        bin_count_2[mask_2] = 0
        count_probs_2 = [i / sum(bin_count_2) for i in bin_count_2]

        entropy = scipy.stats.entropy(count_probs)
        entropy_2 = scipy.stats.entropy(count_probs)

        basic = 0.1
        if np.argmax(logit_sfmean[i, :]) != np.argmax(logit_sfmean_2[i, :]):
            basic = 0.00001
        else:
            basic = (y_inner_confidence[i] + y_inner_confidence_2[i])/2
        between_var = (logit_sfmean[i, :] - logit_sfmean_2[i, :])
        between_var_norm = np.linalg.norm(between_var, ord=2)



        # inner_betw_weighted_entropy = entropy * logit_sfvar[i] / y_inner_confidence[i]
        inner_betw_weighted_entropy = between_var_norm/basic #entropy  #1/y_inner_confidence[i]
        entropy_list.append(inner_betw_weighted_entropy)
    return entropy_list

def drop_entropy_emnlp2(y_prbos, mask_num, logit_sfmean, logit_sfvar, represent_setvar):
    y_probs = np.array(y_prbos)
    entropy_list = []
    entropy_list2 = []
    y_inner_confidence = np.max(logit_sfmean, axis = 1).tolist()
    for i in range(y_probs.shape[1]):
        logit_i = y_probs[:, i]
        bin_count = np.bincount(logit_i)
        mask = sorted(range(len(bin_count)), key=lambda i: bin_count[i])[:-mask_num]
        bin_count[mask] = 0
        count_probs = [i / sum(bin_count) for i in bin_count]

        entropy = scipy.stats.entropy(count_probs)
        # inner_betw_weighted_entropy = entropy * logit_sfvar[i] / y_inner_confidence[i]
        inner_betw_weighted_entropy = 1 / y_inner_confidence[i]
        entropy_list.append(entropy)
        entropy_list2.append(inner_betw_weighted_entropy)
    return entropy_list, entropy_list2

##above is added by 
def uncertain_score2(feature, model, drop_num=20, mask_num=5):
    # dropout
    model.train()
    logit_probs = []
    logit_score_list = []
    y_probs = []

    logit_sfset = []
    represent_set = []

    for i in range(drop_num):
        logit, represent = model(feature)
        logit = torch.nn.Softmax(dim=1)(logit)

        # if i == 0:
        #     logit_mean = torch.zeros_like(logit.detach())
        # logit_mean += logit.detach() * (1/drop_num)

        represent_set.append(represent.cpu().detach().numpy())
        #     # print(repeatTime)
        #     logit, represent = model(feature)
        logit_sfmx = torch.nn.Softmax(dim=1)(logit)
        #     y_pred = (torch.max(logit, 1)[1].view(feature.size()[0]).data).tolist()
        #     prob += y_pred
        logit_sfset.append(logit_sfmx.cpu().detach().numpy())



        # logit_var = np.var(logit.cpu().data.numpy(), axis=1).tolist()      #instance间的准确度
        # logit_s = logit_score(logit, top_k=3)                              #instance内的准确度
        y_pred = (torch.max(logit, 1)[1].view(feature.size()[0]).data).tolist()

        # logit_probs += [logit_var]
        y_probs += [y_pred]
        # logit_score_list += [logit_s]
    # drop_score = drop_entropy(y_probs, mask_num)       #只考虑个数，未考虑 instance间 和 内的准确度和

    logit_sfset = np.stack(np.array(logit_sfset), axis = 2)
    represent_set = np.stack(np.array(represent_set), axis = 2)
    logit_sfmean = logit_sfset.mean(axis=2)
    logit_sfvar = logit_sfset.var(axis=2).sum(axis=1)
    represent_setmean = represent_set.mean(axis=2)            #其实这个地方也可以继续考虑
    represent_setvar = represent_set.var(axis=2).sum(axis=1)
    # inst_weight = 1 - (torch.nn.Sigmoid()(logit_sfvar) - 0.5)*2

    drop_score, drop_score2 = drop_entropy_emnlp2(y_probs, mask_num, logit_sfmean, logit_sfvar, represent_setvar)

    predictive_mean = np.mean(y_probs, axis=0)
    predictive_variance = np.var(y_probs, axis=0)
    #tau = l ** 2 * (1 ­ model.p) / (2 * N * model.weight_decay)
    #predictive_variance += tau **­1
    model.eval()

    # logit
    # logit, represent = model(feature)

    return drop_score, drop_score2, torch.from_numpy(logit_sfmean).float().cuda()
    #return predictive_mean.tolist()
    #return predictive_variance.tolist()

def uncertain_score(feature, model, drop_num=20, mask_num=5):
    # dropout
    model.train()
    logit_probs = []
    logit_score_list = []
    y_probs = []
    y_probs_2 = []

    logit_sfset = []
    represent_set = []
    logit_sfset_2 = []
    represent_set_2 = []

    for i in range(drop_num):
        res = model(feature)
        logit, represent = res[0], res[1]
        logit_2, represent_2 = res[3], res[4]

        logit = torch.nn.Softmax(dim=1)(logit)
        logit_2 = torch.nn.Softmax(dim=1)(logit_2)

        # if i == 0:
        #     logit_mean = torch.zeros_like(logit.detach())
        # logit_mean += logit.detach() * (1/drop_num)

        represent_set.append(represent.cpu().detach().numpy())
        represent_set_2.append(represent_2.cpu().detach().numpy())
        #     # print(repeatTime)
        #     logit, represent = model(feature)
        logit_sfmx = torch.nn.Softmax(dim=1)(logit)
        logit_sfmx_2 = torch.nn.Softmax(dim=1)(logit_2)
        #     y_pred = (torch.max(logit, 1)[1].view(feature.size()[0]).data).tolist()
        #     prob += y_pred
        logit_sfset.append(logit_sfmx.cpu().detach().numpy())
        logit_sfset_2.append(logit_sfmx_2.cpu().detach().numpy())



        # logit_var = np.var(logit.cpu().data.numpy(), axis=1).tolist()      #instance间的准确度
        # logit_s = logit_score(logit, top_k=3)                              #instance内的准确度
        y_pred = (torch.max(logit, 1)[1].view(feature.size()[0]).data).tolist()
        y_pred_2 = (torch.max(logit_2, 1)[1].view(feature.size()[0]).data).tolist()

        # logit_probs += [logit_var]
        y_probs += [y_pred]
        y_probs_2 += [y_pred_2]
        # logit_score_list += [logit_s]
    # drop_score = drop_entropy(y_probs, mask_num)       #只考虑个数，未考虑 instance间 和 内的准确度和

    logit_sfset = np.stack(np.array(logit_sfset), axis = 2)
    represent_set = np.stack(np.array(represent_set), axis = 2)
    logit_sfmean = logit_sfset.mean(axis=2)
    logit_sfvar = logit_sfset.var(axis=2).sum(axis=1)
    represent_setmean = represent_set.mean(axis=2)            #其实这个地方也可以继续考虑
    represent_setvar = represent_set.var(axis=2).sum(axis=1)

    logit_sfset_2 = np.stack(np.array(logit_sfset_2), axis = 2)
    represent_set_2 = np.stack(np.array(represent_set_2), axis = 2)
    logit_sfmean_2 = logit_sfset_2.mean(axis=2)
    logit_sfvar_2 = logit_sfset_2.var(axis=2).sum(axis=1)
    represent_setmean_2 = represent_set_2.mean(axis=2)            #其实这个地方也可以继续考虑
    represent_setvar_2 = represent_set_2.var(axis=2).sum(axis=1)

    # inst_weight = 1 - (torch.nn.Sigmoid()(logit_sfvar) - 0.5)*2
    drop_score = drop_entropy_emnlp(y_probs, mask_num, logit_sfmean, logit_sfvar, represent_setvar, y_probs_2, logit_sfmean_2, logit_sfvar_2, represent_setvar_2)

    predictive_mean = np.mean(y_probs, axis=0)
    predictive_variance = np.var(y_probs, axis=0)
    #tau = l ** 2 * (1 ­ model.p) / (2 * N * model.weight_decay)
    #predictive_variance += tau **­1
    model.eval()

    # logit
    # logit, represent = model(feature)

    return drop_score, torch.from_numpy(0.5 * logit_sfmean + 0.5 * logit_sfmean_2).float().cuda()
    #return predictive_mean.tolist()
    #return predictive_variance.tolist()

## general evaluation
def drop_entropy_eval(dataset, x_val, y_val, model, args):
    print("using drop entropy evaluation ...")
    model.eval()
    y_pred = []
    y_truth = []
    uncertain_score_list = []
    uncertain_score_list2 = []
    represent_all = torch.FloatTensor()
    target_all = torch.LongTensor()
    val_iter = dataset.gen_minibatch(x_val, y_val, args.batch_size, args, shuffle=True)
    repr_output = []
    for batch in val_iter:
        feature, target = batch[0], batch[1]
        #feature.data.t_(), target.data.sub_(1)  # batch first, index align
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        # logit, represent = model(feature)
        

        if args.individual_eval:
            score, logit_mean = uncertain_score(feature, model, args.drop_num, args.drop_mask)
            uncertain_score_list += score
        else:
            score, score2, logit_mean = uncertain_score2(feature, model, args.drop_num, args.drop_mask)
            uncertain_score_list += score
            uncertain_score_list2 += score2





        y_pred_cur = (torch.max(logit_mean, 1)[1].view(target.size()).data).tolist()    #只用了当前的数据，未用平均值
        y_truth_cur = target.data.tolist()

        y_pred += y_pred_cur
        y_truth += y_truth_cur
        # corrects += (torch.max(logit, 1)
        #              [1].view(target.size()).data == target.data).sum()

        # represent_all = torch.cat([represent_all, represent.data.cpu()], 0)
        target_all = torch.cat([target_all, target.data.cpu()], 0)

        # print("\n=== logit ===")
        # print(logit)
        # print("=== prediction ===")
        # print(y_pred_cur)
        # print("=== truth ===")
        # print(y_truth_cur)

   # with open('./snapshot/2018-12-15_03-50-46/pred.csv','w', encoding='utf-8-sig') as f:     
   #     wp = csv.writer(f, delimiter=',')     
   #     wp.writerow(y_pred)
   # with open('./snapshot/2018-12-15_03-50-46/truth.csv', 'w', encoding='utf-8-sig') as f:
   #     wt = csv.writer(f, delimiter=',')
   #     wt.writerow(y_truth)

    with open('./snapshot/DropEntropyEval/pred.txt', 'w', encoding='utf-8-sig') as f:
        for val in y_pred:         
            f.write(str(val) + '\n')

    with open('./snapshot/DropEntropyEval/truth.txt', 'w', encoding='utf-8-sig') as f:
        for val in y_truth:
            f.write(str(val) + '\n')

    with open('./snapshot/DropEntropyEval/label.txt', 'w', encoding='utf-8-sig') as f:
        for val in y_val:
            f.write(str(val) + '\n')

    with open('./snapshot/DropEntropyEval/score.txt', 'w', encoding='utf-8-sig') as f:
        for val in uncertain_score_list:             
            f.write(str(val) + '\n')

    if args.output_repr:
        repr_np = represent_all.numpy()
        for i, repr in enumerate(repr_np):
            repr_str = str(target_all[i])
            for dim in repr:
                repr_str += '\t' + "%.3f" % dim
            repr_output.append(repr_str)
        with open('./output_repr.txt', 'w') as f:
            for repr_str in repr_output:
                f.write(repr_str + '\n')


    # apply idk ratio to filter out the uncertain instances.
    # for i in range(len(y_pred)):
    #     print(uncertain_score_list[i], y_pred[i], y_truth[i],sep='\t')

    if args.use_idk:
        if args.individual_eval:
            indices, L_sorted = zip(*sorted(enumerate(uncertain_score_list), key=itemgetter(1), reverse=False))
        else:
            indices, L_sorted = zip(*sorted(enumerate(uncertain_score_list), key=itemgetter(1), reverse=False)) #希望从大到小排列
            indices2, L_sorted2 = zip(*sorted(enumerate(uncertain_score_list2), key=itemgetter(1), reverse=False))
            pos_ref_score = list(range(0, len(uncertain_score_list)))
            pos_dict = dict(zip(indices, pos_ref_score))
            pos_dict2 = dict(zip(indices2, pos_ref_score))
            position_socre = list(range(0, len(uncertain_score_list))) #希望从小到大排列
            for i in range(len(uncertain_score_list)):
                position_socre[i] = pos_dict[i] + pos_dict2[i]
            indices, L_sorted = zip(*sorted(enumerate(position_socre), key=itemgetter(1), reverse=False))



        idk_list = np.arange(0, 0.45, 0.05)
        for idk_ratio in idk_list:
            #print("=== idk_ratio: ", idk_ratio, " ===")
            test_num = int(len(L_sorted) * (1 - idk_ratio))
            indices_cur = list(indices[:test_num])

            y_truth_cur = [y_truth[i] for i in indices_cur]
            y_pred_cur = [y_pred[i] for i in indices_cur]
            f1_score = show_results(dataset, y_truth_cur, y_pred_cur, represent_all, target_all)

    if args.use_human_idk:
        indices, L_sorted = zip(*sorted(enumerate(uncertain_score_list), key=itemgetter(1), reverse=False))
        idk_list = np.arange(0, 1.05, 0.05)
        for idk_ratio in idk_list:
            # print("=== idk_ratio: ", idk_ratio, " ===")
            test_num = int(len(L_sorted) * (1 - idk_ratio))
            indices_cur = list(indices[:test_num])
            y_truth_cur = [y_truth[i] for i in indices_cur]
            y_pred_cur = [y_pred[i] for i in indices_cur]

            human_indices = list(indices[test_num:])
            y_human = [y_truth[i] for i in human_indices]
            y_truth_cur = y_truth_cur + y_human
            y_pred_cur = y_pred_cur + y_human

            f1_score = show_results(dataset, y_truth_cur, y_pred_cur, represent_all, target_all)

    else:
        f1_score = show_results(dataset, y_truth, y_pred, represent_all, target_all)

    return f1_score



