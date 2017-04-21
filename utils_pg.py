# -*- coding: utf-8 -*-
#pylint: skip-file
import numpy as np
from numpy.random import random as rand
import theano
import theano.tensor as T
import cPickle as pickle
import sys
import os
import shutil
from copy import deepcopy
import theano.sandbox.cuda

# set use gpu programatically
def use_gpu(gpu_id):
    if gpu_id > -1:
        theano.sandbox.cuda.use("gpu" + str(gpu_id))

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_normal_weight(shape, scale=0.01):
    return np.random.normal(loc=0.0, scale=scale, size=shape)

def init_uniform_weight(shape):
    return np.random.uniform(-0.1, 0.1, shape)

def init_xavier_weight_uniform(shape):
    return np.random.uniform(-np.sqrt(6. / (shape[0] + shape[1])), np.sqrt(6. / (shape[0] + shape[1])), shape)

def init_xavier_weight(shape):
    fan_in, fan_out = shape
    s = np.sqrt(2. / (fan_in + fan_out))
    return init_normal_weight(shape, s)

def init_ortho_weight(shape):
    W = np.random.normal(0.0, 1.0, (shape[0], shape[0]))
    u, s, v = np.linalg.svd(W)
    return u

def init_weights(shape, name, sample = "xavier", num_concatenate = 1, axis_concatenate = -1):
    if sample == "uniform":
        if num_concatenate == 1:
            values = init_uniform_weight(shape)
        elif num_concatenate > 1:
            l = []
            for i in range(num_concatenate):
                l.append(init_uniform_weight(shape))
            values = np.concatenate(l, axis = axis_concatenate)
        else:
            raise RuntimeError("Wrong num_concatenate:" + str(num_concatenate))

    elif sample == "normal":
        if num_concatenate == 1:
            values = init_normal_weight(shape)
        elif num_concatenate > 1:
            l = []
            for i in range(num_concatenate):
                l.append(init_normal_weight(shape))
            values = np.concatenate(l, axis = axis_concatenate)
        else:
            raise RuntimeError("Wrong num_concatenate:" + str(num_concatenate))

    elif sample == "xavier":
        if num_concatenate == 1:
            values = init_xavier_weight(shape)
        elif num_concatenate > 1:
            l = []
            for i in range(num_concatenate):
                l.append(init_xavier_weight(shape))
            values = np.concatenate(l, axis = axis_concatenate)
        else:
            raise RuntimeError("Wrong num_concatenate:" + str(num_concatenate))

    elif sample == "ortho":
        if num_concatenate == 1:
            values = init_ortho_weight(shape)
        elif num_concatenate > 1:
            l = []
            for i in range(num_concatenate):
                l.append(init_ortho_weight(shape))
            values = np.concatenate(l, axis = axis_concatenate)
        else:
            raise RuntimeError("Wrong num_concatenate:" + str(num_concatenate))

    else:
        raise ValueError("Unsupported initialization scheme: %s" % sample)

    return theano.shared(floatX(values), name)

def init_gradws(shape, name):
    return theano.shared(floatX(np.zeros(shape)), name)

def init_bias(size, name, num_concatenate = 1):
    if num_concatenate >= 1:
        values = np.zeros((size * num_concatenate,))
    else:
        raise RuntimeError("Wrong num_concatenate:" + str(num_concatenate))
    return theano.shared(floatX(values), name)

def init_real_num(name):
    return theano.shared(rand(), name)

def rebuild_dir(path):
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
        except OSError:
            pass
    os.mkdir(path)

def save_model(f, model):
    ps = {}
    for p in model.params_dis + model.params_gen:
        ps[p.name] = p.get_value()
    pickle.dump(ps, open(f, "wb"), protocol = pickle.HIGHEST_PROTOCOL)

def load_model(f, model):
    ps = pickle.load(open(f, "rb"))
    for p in model.params_dis + model.params_gen:
        p.set_value(ps[p.name])
    return model

def check_nan(x):
    b = np.isnan(x).flatten().tolist()
    for e in b:
        if e:
            print "is nan"
            return True

    print "is not nan"
    return False

def write_tensor3(path, tensor):
    with file(path, "w") as f_dst:
        f_dst.write("# Array shape: {0}\n".format(tensor.shape))
        for i in tensor:
            np.savetxt(f_dst, i)
            f_dst.write("# New Slice\n")


