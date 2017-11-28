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
latent_size = 2
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


print "lode model..."
load_model("./model/gan_text.model", model)


top_w = 21
## manifold 
if latent_size == 3:
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
    z0 = model.noiser(latent_size)
    for i in np.arange(-1, 1, 0.1):
        z = z0 + i
        y = model.generate(z)[0,:]
        ind = np.argsort(-y)
        for k in xrange(top_w):
            print i2w[ind[k]],
        print "\n"
