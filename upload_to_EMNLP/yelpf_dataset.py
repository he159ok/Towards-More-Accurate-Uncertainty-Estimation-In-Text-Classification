# -- coding: utf-8 --
import nltk
from sklearn.datasets import fetch_20newsgroups

import platform
import re
import os
import random
import tarfile
import urllib
import pickle
import numpy as np
import itertools

import torch
from torch.autograd import Variable
#from torchtext import data
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
import os.path
# import torchtext
from nltk import word_tokenize


class Yelpf_Dataset:

    def __init__(self, args):
        """Create an AutoGSR dataset instance. """

        #self.word_embed_file = self.data_folder + 'embedding/wiki.ar.vec'
        # word_embed_file = data_folder + "embedding/Wiki-CBOW"
        self.data_dir = args.data_path + 'yelp/'
        self.vocab_file = self.data_dir + 'vocabulary.txt'
        # self.train_dataset, self.test_dataset = torchtext.datasets.YelpReviewFull(root=self.data_dir)
        self.train_df_file = self.data_dir + 'test_df3.pkl'
        self.test_df_file = self.data_dir + 'train_df3.pkl'
        self.lemmatizer = WordNetLemmatizer()
        self.class_num = -1
        pass

    def clean_str(self, string):
        """
        Tokenization/string cleaning.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip()

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab_file(self):
        return self.vocab_file

    def get_class_num(self):
        return self.class_num

    def cal_class_num(self, df):
        ids_labels = set()

        # since sometimes the data will be shuffled in the frame
        # during train test split
        for index in df.index:
            # labels
            ids_labels.add(df.Class[index])
        return len(ids_labels)

    def load_vocab(self):

        # if not os.path.isfile(self.vocab_size):
        #     # generate the vocab file
        #     newsgroups = fetch_20newsgroups(remove=('headers'))
        #
        #     pass

        with open(self.vocab_file, encoding='utf-8') as f:
            vocab_words = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        vocab_words = [x.strip() for x in vocab_words]

        self.vocab_size = len(vocab_words)
        vocab_wordidx = {w: i for i, w in enumerate(vocab_words)}

        return vocab_wordidx

    def sent_parse(self, text):
        text = text.lower()
        text = text.replace(r'[^A-Za-z0-9 ]+', '')
        text = text.replace(r"\s\s+",' ')

        sentences = nltk.tokenize.sent_tokenize(text)
        sentences_tokens = []

        for sentence in sentences:
            tokens = nltk.tokenize.wordpunct_tokenize(sentence)
            processed_tokens = []
            for token in tokens:
                token = self.clean_str(token)
                token = self.lemmatizer.lemmatize(token)
                if not token:
                    processed_tokens.append(token)
            sentences_tokens.append(tokens)
        return sentences_tokens

    def generate_data(self, val_ratio=.1, shuffle=True):
        print("generating YelpFull data ...")
        train_df, test_df, vocab_wordidx = self.load_data_if_not_exist()

        # if use_small:
        #     train_df = train_df[0:1000]
        #     test_df = test_df[0:500]

        self.class_num = self.cal_class_num(train_df)

        train_df, val_df = train_test_split(train_df,
                                            test_size=val_ratio, random_state=967898,
                                            stratify=train_df.Class)

        x_train, y_train = self.format_data(train_df, vocab_wordidx)
        x_val, y_val = self.format_data(val_df, vocab_wordidx)
        x_test, y_test = self.format_data(test_df, vocab_wordidx)

        return (x_train, y_train), (x_val, y_val), (x_test, y_test)

    def load_data_if_not_exist(self):
        """Create dataset objects for splits of the MR dataset.

        Arguments:
            args: arguments
            val_ratio: The ratio that will be used to get split validation dataset.
            shuffle: Whether to shuffle the data before split.
        """

        # check the existence of data files

        if os.path.isfile(self.train_df_file) and os.path.isfile(self.test_df_file):
            train_df = pd.read_pickle(self.train_df_file)
            test_df = pd.read_pickle(self.test_df_file)
            vocab_wordidx = self.load_vocab()
            return train_df, test_df, vocab_wordidx

        # load data
        categories = ['alt.atheism',
                      'comp.graphics',
                      'comp.os.ms-windows.misc',
                      'comp.sys.ibm.pc.hardware',
                      'comp.sys.mac.hardware',
                      'comp.windows.x',
                      'misc.forsale',
                      'rec.autos',
                      'rec.motorcycles',
                      'rec.sport.baseball',
                      'rec.sport.hockey',
                      'sci.crypt',
                      'sci.electronics',
                      'sci.med',
                      'sci.space',
                      'soc.religion.christian',
                      'talk.politics.guns',
                      'talk.politics.mideast',
                      'talk.politics.misc',
                      'talk.religion.misc'
                      ]

        #train_categories = categories[0:15]
        train_categories = categories
        # newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers'),
        #                                       categories=train_categories)
        #
        # newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers'))
        train_df = pd.read_pickle(self.train_df_file)
        test_df = pd.read_pickle(self.test_df_file)

        # load vocab_words and vocab_wordidx
        # with open(self.vocab_words_file, 'rb') as f:
        #     (vocab_words, vocab_idx) = pickle.load(f)
        # self.vocab_size = len(vocab_words)
        print('start transfer')
        train_df = pd.DataFrame(
            {'Text': [self.sent_parse(text) for text in train_df.Text],
             'Class': train_df.Class}
        )

        test_df = pd.DataFrame(
            {'Text': [self.sent_parse(text) for text in test_df.Text],
             'Class': test_df.Class}
        )

        self.train_df_file = './data/yelp/train_df3.pkl'
        self.test_df_file = './data/yelp/test_df3.pkl'

        train_df.to_pickle(self.train_df_file)
        test_df.to_pickle(self.test_df_file)

        print('finish individual save')
        return None

        # extract title and sentence words
        train_sent_words = train_df.Text.apply(lambda ll: list(itertools.chain.from_iterable(ll)))
        train_sent_words = list(itertools.chain.from_iterable(train_sent_words))
        test_sent_words = test_df.Text.apply(lambda ll: list(itertools.chain.from_iterable(ll)))
        test_sent_words = list(itertools.chain.from_iterable(test_sent_words))

        # generate vocabulary words
        vocab_words = list(set(train_sent_words) | set(test_sent_words))
        # add extra words such as start/end of sentence
        vocab_words.append("<UNK>")
        vocab_words.append("<SOSent>")
        vocab_words.append("<EOSent>")
        vocab_words.append("<SODoc>")
        vocab_words.append("<EODoc>")

        vocab_words.sort()

        # save vocabulary words and word index into files
        with open(self.vocab_file, 'w') as f:
            for word in vocab_words:
                f.write(word)
                f.write('\n')
            #pickle.dump(vocab_words, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.vocab_size = len(vocab_words)
        vocab_wordidx = {w: i for i, w in enumerate(vocab_words)}

        return train_df, test_df, vocab_wordidx

    def format_data(self, data_frame, vocab_idx):

        ids_document = []
        ids_labels = []

        # since sometimes the data will be shuffled in the frame
        # during train test split
        for index in data_frame.index:
            document = data_frame.Text[index]
            # document = [word_tokenize(data_frame.Text[index])]
            text_word_list = [self.convertSent2WordIds(sentence, vocab_idx) for sentence in
                              document]
            #text_word_list = [j for j in i for i in text_word_list]
            text_word_list = [item for sublist in text_word_list for item in sublist]
            ids_document.append(text_word_list)

            # labels
            ids_labels.append(data_frame.Class[index] - 1)

        return np.array(ids_document), np.array(ids_labels)

    def convertSent2WordIds(self, sentence, vocab_idx):
        """
        sentence is a list of word.
        It is converted to list of ids based on vocab_idx
        """
        sentence_start_tag_idx = vocab_idx["<SOSent>"]
        sentence_end_tag_idx = vocab_idx["<EOSent>"]
        word_unknown_tag_idx = vocab_idx["<UNK>"]

        sent2id = [sentence_start_tag_idx]
        #sent2id = []

        try:
            mid = []
            for word in sentence:
                try:
                    if vocab_idx[word] < self.vocab_size:
                        mid.append(vocab_idx[word])
                    else:
                        mid.append(word_unknown_tag_idx)
                except:
                    # mid.append(word_unknown_tag_idx)
                    continue
            sent2id = sent2id + mid

            # sent2id = sent2id + [
            #     vocab_idx[word] if vocab_idx[word] < self.vocab_size else word_unknown_tag_idx for
            #     word in sentence]
        except KeyError as e:
            print(e)
            print(sentence)
            raise ValueError('Fix this issue dude')

        sent2id = sent2id + [sentence_end_tag_idx]
        return sent2id

    def iterate_minibatches(self, inputs, targets, batchsize, shuffle=False):
        assert inputs.shape[0] == targets.shape[0]
        if shuffle:
            indices = np.arange(inputs.shape[0])
            np.random.shuffle(indices)
        for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt], excerpt

        # return last part
        start_idx = inputs.shape[0] - int(inputs.shape[0]) % batchsize
        if shuffle:
            excerpt = indices[start_idx:]
        else:
            excerpt = slice(start_idx, inputs.shape[0])
        yield inputs[excerpt], targets[excerpt], excerpt

    def gen_minibatch(self, tokens, labels, mini_batch_size, args, shuffle=True):
        for token, label, excerpt in self.iterate_minibatches(tokens, labels, mini_batch_size, shuffle=shuffle):
            if len(token) == 0:
                continue
            token = self.pad_batch(token)
            token.data.t_()
            label = Variable(torch.from_numpy(label), requires_grad=False)
            if args.cuda == True:
                token, label = token.cuda(), label.cuda()

            yield (token, label, excerpt)

    def pad_batch(self, mini_batch):
        mini_batch_size = len(mini_batch)
        #mean_sent_len = int(np.mean([len(x) for x in mini_batch]))
        mean_token_len = int(np.mean([len(x) for x in mini_batch]))
        max_token_len = int(np.max([len(x) for x in mini_batch]))
        main_matrix = np.zeros((mini_batch_size, mean_token_len), dtype=np.int)
        for i in range(main_matrix.shape[0]):
            for j in range(main_matrix.shape[1]):
                    try:
                        main_matrix[i, j] = mini_batch[i][j]
                    except IndexError:
                        pass
        return Variable(torch.from_numpy(main_matrix).transpose(0, 1))


