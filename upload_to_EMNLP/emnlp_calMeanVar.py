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
from eval import eval
import math

def multiclass_metric_loss(represent, target, margin=10, class_num=2, start_idx=1):
    target_list = target.data.tolist()
    dim = represent.data.shape[1]
    indices = []
    for class_idx in range(start_idx, class_num + start_idx):
        indice_i = [i for i, x in enumerate(target_list) if x == class_idx]
        indices.append(indice_i)

    loss_intra = Variable(torch.FloatTensor([0])).cuda()
    num_intra = 0
    loss_inter = Variable(torch.FloatTensor([0])).cuda()
    num_inter = 0
    for i in range(class_num):
        # intra class loss
        indice_i = indices[i]
        for intra_i in range(len(indice_i)):
            for intra_j in range(intra_i + 1, len(indice_i)):
                r_i = represent[indice_i[intra_i]]
                r_j = represent[indice_i[intra_j]]
                dist_ij = (r_i - r_j).norm(2)
                loss_intra += 1 / dim * (dist_ij * dist_ij)
                num_intra += 1

        # inter class loss
        for j in range(i + 1, class_num):
            indice_j = indices[j]
            for inter_i in indice_i:
                for inter_j in indice_j:
                    r_i = represent[inter_i]
                    r_j = represent[inter_j]
                    dist_ik = (r_i - r_j).norm(2)
                    tmp = margin - 1 / dim * (dist_ik * dist_ik)
                    loss_inter += torch.clamp(tmp, min=0)
                    num_inter += 1
    if num_intra > 0:
        loss_intra = loss_intra / num_intra
    if num_inter > 0:
        loss_inter = loss_inter / num_inter
    return loss_intra, loss_inter

def metric_loss(represent, target, margin):

    target_list = target.data.tolist()
    dim = represent.data.shape[1]
    indices = [i for i, x in enumerate(target_list) if x == 1]
    other_indices = list(set(range(0, len(target_list))) - set(indices))

    # no label 1 instances
    if len(indices) == 0:
        return Variable(torch.FloatTensor([0])).cuda(), Variable(torch.FloatTensor([0])).cuda()

    loss_intra = Variable(torch.FloatTensor([0])).cuda()
    num_intra = 0
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            r_i = represent[indices[i]]
            r_j = represent[indices[j]]
            dist_ij = (r_i - r_j).norm(2)
            loss_intra += 1 / dim * (dist_ij * dist_ij)
            num_intra += 1
    if num_intra > 0:
        loss_intra = loss_intra / num_intra

    loss_inter = Variable(torch.FloatTensor([0])).cuda()
    num_inter = 0
    for i in indices:
        for k in other_indices:
            r_i = represent[i]
            r_k = represent[k]
            dist_ik = (r_i - r_k).norm(2)
            tmp = margin - 1 / dim * (dist_ik * dist_ik)
            loss_inter += torch.clamp(tmp, min=0)
            num_inter += 1
    if num_inter > 0:
        loss_inter = loss_inter / num_inter
    return loss_intra, loss_inter

def ramp_up_function(epoch, epoch_with_max_rampup=80):
    """ Ramps the value of the weight and learning rate according to the epoch
        according to the paper
    Arguments:
        {int} epoch
        {int} epoch where the rampup function gets its maximum value
    Returns:
        {float} -- rampup value
    """

    if epoch < epoch_with_max_rampup:
        p = max(0.0, float(epoch)) / float(epoch_with_max_rampup)
        p = 1.0 - p
        return math.exp(-p*p*5.0)
    else:
        return 1.0

def CalMeanVar(dataset, x_train, y_train, x_val, y_val, model, args):
    if args.cuda:
        print("training model in cuda ...")
        model.cuda()

    flag = False
    if args.mixup:
        args.mixup = False
        flag = True

    print('training using CNN Model Type', args.model_type, ' ...')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps = 0
    best_acc = 0
    last_step = 0
    classNum = args.class_num
    repeatModelNums = 5
    embedDimNum = 300
    Z = torch.zeros((x_train.shape[0], classNum))
    z = torch.zeros((x_train.shape[0], classNum))
    sample_epoch = torch.zeros((x_train.shape[0], 1))
    alpha = 0.6
    max_unsupervised_weight = 20 #50

    model.train()
    print(model)
    for epoch in range(1):
        rampup_value = ramp_up_function(epoch, 40)
        print("epoch", epoch, "rampup_value", rampup_value)
        representSSet = []
        targetSSet = []
        if epoch == 0:
            unsupervised_weight = 0
        else:
            unsupervised_weight = max_unsupervised_weight * \
                                  rampup_value

        train_iter = dataset.gen_minibatch(x_train, y_train, args.batch_size, args, shuffle=True) #args.batch_size
        for batch in train_iter:
            feature, target, excerpt = batch[0], batch[1], batch[2]
            batch_size = feature.shape[0]
            target_onehot = torch.zeros(feature.shape[0], args.class_num).scatter_(1, target.unsqueeze(1).cpu(), 1)
            #feature.data.t_(), target.data.sub_(1)  # batch first, index align
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

                target_onehot = target_onehot.cuda()
            optimizer.zero_grad()

            current_ensemble_indexes = excerpt
            current_ensemble_targets = z[current_ensemble_indexes]

            if args.mixup:
                args.mixup = False
                # res = model(feature.float(), target_onehot, batch_size) #for xlnet
                res = model(feature, target_onehot, batch_size)
                logit_sfmx, represent = res[0], res[1]
                representSSet.append(represent.detach().cpu())
                targetSSet.extend(target.cpu().numpy())

            if flag:
                args.mixup = True

    print('finish model calculation')
    processedData_Ori = torch.cat(representSSet, dim = 0)
    #根据类别算均值，建立 target，根据整体算协方差
    processedData = processedData_Ori.t()
    processedData = processedData.numpy()
    cov_mat = np.cov(processedData)
    cov_mat_inv = np.linalg.inv(cov_mat)
    targetSSet = np.array(targetSSet)
    meanSSet = np.zeros((args.class_num, cov_mat.shape[0]))
    cov_mat_inv_class = np.zeros((args.class_num, cov_mat.shape[0], cov_mat.shape[0]))

    for classID in range(args.class_num):
        classArr = np.where(targetSSet == classID)
        mid = processedData_Ori[classArr[0], :].numpy()
        classMean = np.mean(mid, axis=0)   #可以考虑对mid加入log_softmax
        meanSSet[classID, :] = classMean
        # mid_cov_mat = np.cov(mid.T)
        # mid_cov_mat_inv = np.linalg.inv(mid_cov_mat)
        # cov_mat_inv_class[classID, :, :] = mid_cov_mat_inv


    print("start saving data")
    Mdict = {'global_cov_inv':cov_mat_inv, 'local_mean': meanSSet}
    if args.dataset == "20news":
        filename = '20news_Mdistance_helper.npy'
    elif args.dataset == "amazon":
        filename = 'amazon_Mdistance_helper.npy'
    elif args.dataset == "yelp":
        filename = 'yelp_Mdistance_helper.npy'
    elif args.dataset == "imdb":
        filename = 'imdb_Mdistance_helper.npy'
    elif args.dataset == "amazon_xlnet":
        filename = 'amazonXLNET_Mdistance_helper.npy'
    print(filename)
    np.save(filename, Mdict)
    print("finish saving data")





















            # if steps % args.save_interval == 0:
            #     save(model, args.save_dir, 'snapshot', steps)


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)