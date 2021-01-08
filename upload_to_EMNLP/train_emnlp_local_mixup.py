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

def train(dataset, x_train, y_train, x_val, y_val, model, args):
    if args.cuda:
        print("training model in cuda ...")
        model.cuda()

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
    for epoch in range(1, args.epochs+1):
        rampup_value = ramp_up_function(epoch, 40)
        print("epoch", epoch, "rampup_value", rampup_value)

        if epoch == 0:
            unsupervised_weight = 0
        else:
            unsupervised_weight = max_unsupervised_weight * \
                                  rampup_value

        train_iter = dataset.gen_minibatch(x_train, y_train, args.batch_size, args, shuffle=True)
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


            #below add by 
            # logit_sfset = []
            # prob = []
            # represent_set = []
            # for repeatTime in range(repeatModelNums):
            #     # print(repeatTime)
            #     logit, represent = model(feature)
            #     logit_sfmx = torch.nn.Softmax(dim=1)(logit)
            #     y_pred = (torch.max(logit, 1)[1].view(feature.size()[0]).data).tolist()
            #     prob += y_pred
            #     logit_sfset.append(logit_sfmx)
            #     represent_set.append(represent)
            # logit_sfset = torch.stack(logit_sfset, dim = 2)
            # represent_set = torch.stack(represent_set, dim = 2)
            # logit_sfmean = logit_sfset.mean(dim=2)
            # logit_sfvar = logit_sfset.var(dim=2).sum(dim=1)
            # represent_setmean = represent_set.mean(dim=2)
            # represent_setvar = represent_set.var(dim=2).sum(dim=1)
            # inst_weight = 1 - (torch.nn.Sigmoid()(logit_sfvar) - 0.5)*2

            #y_pred_sfmean = (torch.max(logit_sfmean, 1)[1].view(feature.size()[0]).data).tolist()
            #above add by 

            # 单dropout下的类内间的使用
            if args.mixup:
                res = model(feature, target_onehot, batch_size)
                logit_sfmx, represent, target_onehot_processed = res[0], res[1], res[2]
                # loss_target = F.cross_entropy(logit_sfmx, target)
                loss_target = torch.nn.KLDivLoss(reduce='batchmean')(F.log_softmax(logit_sfmx, 1), target_onehot_processed)
                            # + torch.nn.L1Loss()((F.cross_entropy(target_onehot_processed, target, reduce=False)), (F.cross_entropy(logit_sfmx, target, reduce=False)))
                              # + torch.max(input = (F.cross_entropy(target_onehot_processed, target, reduce=False) - F.cross_entropy(logit_sfmx, target, reduce=False)), other=(Variable(torch.zeros_like(target).float()))).sum()
                              # + F.cross_entropy(logit_sfmx, target)
                # target_onehot * F.log_softmax(target_onehot_processed, 1)

            else:
                res = model(feature)
                logit_sfmx, represent = res[0], res[1]
                loss_target = F.cross_entropy(logit_sfmx, target)
            # logit_sfmx = Variable(torch.nn.Softmax(dim=1)(logit))
            # inst_weight = torch.max(logit_sfmx, 1)[0] - torch.min(logit_sfmx, 1)[0]

            # loss_target = F.cross_entropy(logit_sfmx, target)
            # loss_target = torch.nn.KLDivLoss(reduce='batchmean')(F.log_softmax(logit_sfmx, 1), target_onehot)
            # loss_target = torch.nn.KLDivLoss(reduce='mean')(logit_sfmx, target_onehot) 没有logSfmx的情况，梯度学不起来
            # loss_target = F.mse_loss(logit_sfmx, target_onehot)
            if args.selfensemble:
                logit_sfmx_2, represent_2 = res[3], res[4]
                loss_between = F.mse_loss(logit_sfmx, logit_sfmx_2)
                loss_target2 = torch.nn.KLDivLoss(reduce='batchmean')(F.log_softmax(logit_sfmx_2, 1), target_onehot_processed)
            else:
                loss_between = F.mse_loss(logit_sfmx, current_ensemble_targets.cuda())  #时序的近似版本

            inst_weight = F.mse_loss(logit_sfmx, torch.zeros_like(logit_sfmx))#(torch.max(logit_sfmx, 1)[0] - torch.min(logit_sfmx, 1)[0])
            inst_weight_show = (torch.max(logit_sfmx, 1)[0] - torch.min(logit_sfmx, 1)[0])
            loss_var = -1 * inst_weight_show.sum()


            #loss_target = torch.nn.CrossEntropyLoss(weight=inst_weight, reduction='none')(logit_sfmean, target)
            # loss_target = Variable(torch.FloatTensor([0])).cuda()
            # for i in range(target.shape[0]):
            #     if i == 0:
            #         loss_target += F.cross_entropy(logit_sfmean[i,:].unsqueeze(0), torch.cuda.LongTensor([target[i]])) * inst_weight[i]
            #     else:
            #         loss_target +=  F.cross_entropy(logit_sfmean[i,:].unsqueeze(0), torch.cuda.LongTensor([target[i]])) * inst_weight[i]



            loss_metric = Variable(torch.FloatTensor([0])).cuda()
            if args.metric:
                class_num = dataset.get_class_num()
                if class_num == 2:
                    #loss_intra, loss_inter = metric_loss(represent, target, margin=args.metric_margin)
                    loss_intra, loss_inter = multiclass_metric_loss(represent, target, margin=args.metric_margin,
                                                                    class_num=class_num, start_idx=0)
                elif class_num > 2:
                    loss_intra, loss_inter = multiclass_metric_loss(represent, target, margin=args.metric_margin, class_num=class_num)
                # try:
                #
                # except:
                #     loss_intra, loss_inter = metric_loss(represent, target, margin=10)
                #     a = 0
                loss_metric = loss_intra + loss_inter

            #loss_var = torch.sum(logit_sfvar)
            # loss_var = -1 * inst_weight.sum()
            if args.selfensemble:
                loss = loss_target +  loss_target2 + args.intraRate * loss_between #+ unsupervised_weight * loss_between
            else:
                loss = loss_target  #+  args.metric_param * loss_metric #+ 0.1 * inst_weight # + unsupervised_weight * loss_between
            # loss = loss_target  + -1 * unsupervised_weight * loss_between + unsupervised_weight * loss_var

            #print('logit vector', logit.size())
            #print('target vector', target.size())
            loss.backward()
            optimizer.step()


            Z[current_ensemble_indexes, :] = alpha * \
                                             Z[current_ensemble_indexes, :] + (1 - alpha) * logit_sfmx.cpu()
            z[current_ensemble_indexes, :] = Z[current_ensemble_indexes, :] * \
                                             (1. / (1. - alpha **
                                                    (sample_epoch[current_ensemble_indexes] + 1)))
            sample_epoch[current_ensemble_indexes] += 1

            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit_sfmx, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects / args.batch_size
                try:
                    if args.metric:
                        try:
                            sys.stdout.write(
                            '\rEpoch[{}] Batch[{}] - loss: {:.4f}'
                            '({:.6f}/{:.4f}/{:.5f}/{:.5f}) acc: {:.4f}%({}/{})'.format(epoch,
                                                                         steps,
                                                                         loss.data,
                                                                         loss_target.data,
                                                                         loss_metric.data[0],
                                                                         loss_between.data,
                                                                         loss_var.data,
                                                                         accuracy,
                                                                         corrects,
                                                                         args.batch_size))

                        except:
                            sys.stdout.write(
                            '\rEpoch[{}] Batch[{}] - loss: {:.4f}'
                            '({:.6f}/{:.4f}/{:.5f}/{:.5f}) acc: {:.4f}%({}/{})'.format(epoch,
                                                                                       steps,
                                                                                       loss.data[0],
                                                                                       loss_target.data,
                                                                                       loss_metric.data[0],
                                                                                       loss_between.data,
                                                                                       loss_var.data,
                                                                                       accuracy,
                                                                                       corrects,
                                                                                       args.batch_size))
                    else:
                        sys.stdout.write(
                            '\rEpoch[{}] Batch[{}] - loss: {:.4f} '
                            'acc: {:.4f}%({}/{})'.format(epoch,
                                                         steps,
                                                         loss.data[0],
                                                         accuracy,
                                                         corrects,
                                                         args.batch_size))
                except:
                    print("Unexpected error:", sys.exc_info()[0])

            if steps % args.test_interval == 0:

                dev_acc = eval(dataset, x_val, y_val, model, args)  # F1 Score

                if epoch > 1 and epoch % 10 == 0:
                    save(model, args.save_dir, 'epoch', epoch)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
                model.train()

            # if steps % args.save_interval == 0:
            #     save(model, args.save_dir, 'snapshot', steps)


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)