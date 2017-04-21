# -*- coding: utf-8 -*-
#pylint: skip-file
import sys
import os
import numpy as np
import theano
import theano.tensor as T
import cPickle, gzip
import string

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))

def load_stopwords():
    stop_words = {}
    f = open("./data/stopwords.txt", "r")
    for line in f:
        line = line.strip('\n').strip()
        stop_words[line] = 1
    return stop_words

def apnews():
    dic = {}
    i2w = {}
    w2i = {}
    docs = {}
    stop_words = load_stopwords()
    
    f = open("./data/news_ap.txt", "r")
    doc_id = 0
    for line in f:
        line = line.strip('\n').lower()
        line = line.translate(None, string.punctuation)
        words = line.split()
        d = []
        for w in words:
            if w in stop_words:
                continue
            d.append(w)
            if w in dic:
                dic[w] += 1
            else:
                dic[w] = 1
                w2i[w] = len(i2w)
                i2w[len(i2w)] = w

        docs[doc_id] = d
        doc_id += 1
    f.close()

    print len(docs), len(w2i), len(i2w), len(dic)
    doc_idx = [i for i in xrange(len(docs))]
    spliter = (int) (len(docs) / 10.0 * 9)
    train_idx = doc_idx[0:spliter]
    valid_idx = doc_idx[spliter:len(docs)]
    test_idx = valid_idx

    return train_idx, valid_idx, test_idx, [docs, dic, w2i, i2w]

def batched_idx(lst, batch_size = 1):
    np.random.shuffle(lst)
    data_xy = {}
    batch_x = []
    batch_id = 0
    for i in xrange(len(lst)):
        batch_x.append(lst[i])
        if (len(batch_x) == batch_size) or (i == len(lst) - 1):
            data_xy[batch_id] = batch_x
            batch_id += 1
            batch_x = []
    return data_xy

def batched_news(x_idx, data):
    [docs, dic, w2i, i2w] = data
    X = np.zeros((len(x_idx), len(dic)), dtype = theano.config.floatX)    
    for i in xrange(len(x_idx)):
        xi = x_idx[i]
        d = docs[xi]
        for w in d:
            X[i, w2i[w]] += 1
  
    for i in xrange(len(x_idx)):
        norm2 = np.linalg.norm(X[i,:])
        #norm2 = np.sum(X[i,:])
        if norm2 != 0:
            X[i,:] /= norm2
    
    return X

