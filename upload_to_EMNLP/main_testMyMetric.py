#! /usr/bin/env python
# -- coding: utf-8 --
import os
import argparse
import datetime
import torch
# import torchtext.data as data
# import torchtext.datasets as datasets
import pandas as pd
import cnn_model
#import train
import dropout_eval
import drop_entropy_eval
import distance_eval
import logit_eval
import autogsr_dataset
import newsgrp_dataset
import numpy as np
import emnlp_eval
import cnn_model_mixup
import selfensemble_eval
import emnlp_calMeanVar

import bert_model_mixup


from autogsr_dataset import AutoGSR_DataSet
from newsgrp_dataset import NewsGroup_DataSet
from imdb_dataset import IMDB_DataSet
from amazon_dataset import Amazon_DataSet
from yelpf_dataset import Yelpf_Dataset
from amazon_dataset_XLnet import Amazon_DataSet_XLnet


parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=100, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=32, help='batch size for training [default: 32]')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=1000, help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data
parser.add_argument('-data-path', type=str, default='./data/', help='the data directory')
parser.add_argument('-dataset', type=str, default='20news', help='choose dataset to run [options: 20news, imdb, amazon, autogsr]')
parser.add_argument('-shuffle', action='store_true', default=True, help='shuffle the data every epoch')

# model
parser.add_argument('-model-type', type=int, default=1, help='different structures of metric model, see document for details')
parser.add_argument('-dropout', type=float, default=0.3, help='the probability for dropout [default: 0.3]')
parser.add_argument('-embed-dropout', type=float, default=0, help='the probability for dropout [default: 0]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=200, help='number of embedding dimension [default: 200]')
parser.add_argument('-glove', type=bool, default=True, help='whether to use Glove pre-trained word embeddings')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# model-metric learning
parser.add_argument('-metric', action='store_true', default=False, help='use the metric learning')
parser.add_argument('-metric-param', type=float, default=0.1, help='the parameter for the loss of metric learning [default: 0.1]')
parser.add_argument('-metric-margin', type=float, default=100, help='the parameter for margin between different classes [default: 10]')
# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# evaluation
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-small', type=bool, default=False, help='use the regular data or small data')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
parser.add_argument('-openev', action='store_true', default=False, help='use the open class for testing')
parser.add_argument('-dropev', action='store_true', default=False, help='use the dropout bayesian method for uncertainty testing')
parser.add_argument('-dropentev', action='store_true', default=False, help='use the dropout bayesian method based on logit layer for uncertainty testing')
parser.add_argument('-drop-mask', type=int, default=5, help='the number of masks used for dropout bayesian method [default: 5]')
parser.add_argument('-drop-num', type=int, default=100, help='the number of the experiments used for dropout bayesian method [default: 100]')
parser.add_argument('-distev', action='store_true', default=False, help='use the distance method for uncertainty testing')
parser.add_argument('-logitev', action='store_true', default=False, help='use the logit difference for uncertainty testing')
parser.add_argument('-logitev-topk', type=int, default=5, help='the topk parameter for the loss of metric learning [default: 5]')
parser.add_argument('-idk_ratio', type=float, default=0, help='the ratio of uncertainty')
parser.add_argument('-use_idk', action='store_true', default=False, help='use idk. If yes, it will show all the results from 0 to 0.4 with interval 0.05')
parser.add_argument('-use_human_idk', action='store_true', default=False, help='use human idk. If yes, it will show all the results from 0 to 0.4 assuming the uncertain part is handed over to humans')
parser.add_argument('-output_repr', action='store_true', default=False, help='output the representation to file output_repr.txt')
parser.add_argument('-emnlp', action='store_true', default=False, help='apply adapt thoughts to calculate confidence')
parser.add_argument('-emnlptev', action='store_true', default=False, help='use the dropout bayesian method for uncertainty testing')
parser.add_argument('-individual_eval', action='store_true', default=False, help='use single socre or two scores to eval')
parser.add_argument('-mixup', action='store_true', default=False, help='if use mixup')
parser.add_argument('-selfensemble', action='store_true', default=False, help='if use self-ensemble')
parser.add_argument('-betweentev', action='store_true', default=False, help='if use self-ensemble tev, cannot exist with emnlptev')
parser.add_argument('-intraRate', type=float, default=0.01, help='the ratio of intra loss')
# parser.add_argument('-mixmetric', action='store_true', default=False, help='if use mixup and metric')
parser.add_argument('-calmeanvar', action='store_true', default=False, help='if use self-ensemble tev, cannot exist with emnlptev')
parser.add_argument('-lowb', type=float, default=0.75, help='the lower boundary of alpha [0.51, 1]')
parser.add_argument('-lstm', action='store_true', default=False, help='true: Bi-Lstm, cannot exist with emnlptev')

args = parser.parse_args()


if args.emnlp:
    # import train_emnlp_local as train
    # import train_emnlp_Inner as train  #结论是他们的类内损失会不间断的被考虑进去
    import train_emnlp_local_mixup as train
    # import train_emnlp_local_mixup_distinguish as train
else:
    import train as train




# update args and print
args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
if not args.cuda:
    print('Using a small dataset for debugging ...')
    args.small = True
    args.drop_num = 20
    args.embed_dim = 50

# load data
if args.dataset == 'autogsr':
    print("\nLoading autogsr data...")
    dataset = AutoGSR_DataSet(args)
elif args.dataset == '20news':
    print("\nLoading 20news data...")
    dataset = NewsGroup_DataSet(args)
elif args.dataset == 'imdb':
    print("\nLoading IMBD data...")
    dataset = IMDB_DataSet(args)
elif args.dataset == 'amazon':
    print("\nLoading Amazon data...")
    dataset = Amazon_DataSet(args)
elif args.dataset == 'yelp':
    print("\nLoading Yelp FUll data...")
    dataset = Yelpf_Dataset(args)
elif args.dataset == 'amazon_xlnet':
    print("\nLoading Amazon_XLnet data...")
    dataset = Amazon_DataSet_XLnet(args)

(x_train, y_train), (x_val, y_val), (x_test, y_test) = dataset.generate_data()
# args.vocab_size = dataset.get_vocab_size()
# args.class_num = dataset.get_class_num()
try:
    args.vocab_size = dataset.get_vocab_size()
except:
    args.vocab_size = "Bert"
try:
    args.class_num = dataset.get_class_num()
except:
    args.class_num = 5

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

# model
if args.mixup:
    cnn = cnn_model_mixup.CNN_Text(args)
    # cnn = bert_model_mixup.Bert_Text(args)
else:
    cnn = cnn_model.CNN_Text(args)
if args.snapshot is not None:
    print('\nLoading model from {}...'.format(args.snapshot))
    cnn.load_state_dict(torch.load(args.snapshot))
    print('load model successfully')

if args.cuda:
    torch.cuda.set_device(args.device)
    cnn = cnn.cuda()

if args.calmeanvar:
    emnlp_calMeanVar.CalMeanVar(dataset, x_train, y_train, x_val, y_val, cnn, args)

print('finish save npy file')





