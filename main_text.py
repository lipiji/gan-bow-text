#pylint: skip-file
import time
import sys
import numpy as np
import theano
import theano.tensor as T
from GAN import *
import data
import matplotlib.pyplot as plt

use_gpu(0)

lr = 0.001
drop_rate = 0.
batch_size = 100
hidden_size = 300
latent_size = 50
iter_d = 1

# try: sgd, momentum, rmsprop, adagrad, adadelta, adam, nesterov_momentum
optimizer = "rmsprop"

train_idx, valid_idx, test_idx, other_data = data.apnews()
[docs, dic, w2i, i2w] = other_data

dim_x = len(dic)
dim_y = dim_x
print "#features = ", dim_x, "#labels = ", dim_y

print "compiling..."
model = GAN(dim_x, hidden_size, latent_size, optimizer)

print "training..."
start = time.time()
for i in xrange(100):
    train_xy = data.batched_idx(train_idx, batch_size)
    error_d = 0.0
    error_g = 0.0
    in_start = time.time()
    for batch_id, x_idx in train_xy.items():
        local_bath_size = len(x_idx)
        X = data.batched_news(x_idx, other_data)
        Z = model.noiser(local_bath_size)
        
        loss_d = 0
        for di in xrange(iter_d):
            loss_d += model.train_d(X, Z, lr)
        loss_d = loss_d / iter_d
        loss_g = model.train_g(X, Z, lr)

        error_d += loss_d
        error_g += loss_g
        #print i, batch_id, "/", len(train_xy), cost
    in_time = time.time() - in_start

    error_d /= len(train_xy);
    error_g /= len(train_xy);
    print "Iter = " + str(i) + ", loss_d = " + str(error_d) \
            + ", loss_g = " + str(error_g) + ", Time = " + str(in_time)

print "training finished. Time = " + str(time.time() - start)

print "save model..."
save_model("./model/gan_text.model", model)

print "lode model..."
load_model("./model/gan_text.model", model)


top_w = 20
## manifold 
if latent_size == 2:
    nx = ny = 20
    v = 100
    x_values = np.linspace(-v, v, nx)
    y_values = np.linspace(-v, v, ny) 
    canvas = np.empty((28*ny, 20*nx))
    for i, xi in enumerate(x_values):
        for j, yi in enumerate(y_values):
            z = np.array([[xi, yi]], dtype=theano.config.floatX)
            y = model.generate(z)[0,:]
            ind = np.argsort(-y)
            print xi, yi, 
            for k in xrange(top_w):
                print i2w[ind[k]],
            print "\n"
else:
    for i in xrange(20):
        z = model.noiser(latent_size)
        y = model.generate(z)[0,:]
        ind = np.argsort(-y)
        for k in xrange(top_w):
            print i2w[ind[k]],
        print "\n"
