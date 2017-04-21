#pylint: skip-file
import numpy as np
import theano
import theano.tensor as T
from utils_pg import *
from updates import *
import matplotlib.pyplot as plt
from matplotlib import animation

class GAN():
    def __init__(self, in_size, hidden_size, latent_size, optimizer):
        self.X = T.matrix("X")
        self.Z = T.matrix("Z")
        self.optimizer = optimizer
        #self.batch_size = T.iscalar('batch_size')
        self.in_size = in_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        
        self.define_train_test_funcs()

    def noiser(self, n):
        z = init_normal_weight((n, self.latent_size))
        return floatX(z)

    class Generator():
        def __init__(self, out_size, latent_size, hidden_size):
            prefix = "gen_"
            self.out_size = out_size
            self.latent_size = latent_size;
            self.hidden_size = hidden_size
            self.Wg_zh = init_weights((self.latent_size, self.hidden_size), prefix + "Wg_zh")
            self.bg_zh = init_bias(self.hidden_size, prefix + "bg_zh")
            self.Wg_hy = init_weights((self.hidden_size, self.out_size), prefix + "Wg_hy")
            self.bg_hy = init_bias(self.out_size, prefix + "bg_hy")
            self.params = [self.Wg_zh, self.bg_zh, self.Wg_hy, self.bg_hy]
        
        def generate(self, z):
            h = T.tanh(T.dot(z, self.Wg_zh) + self.bg_zh)
            y = T.nnet.sigmoid(T.dot(h, self.Wg_hy) + self.bg_hy)
            return y

    class Discriminator():
        def __init__(self, in_size, hidden_size):
            prefix = "dis_"
            self.in_size = in_size
            self.out_size = 1
            self.hidden_size = hidden_size
            self.Wd_xh = init_weights((self.in_size, self.hidden_size), prefix + "Wd_xh")
            self.bd_xh = init_bias(self.hidden_size, prefix + "bd_xh")
            self.Wd_xh2 = init_weights((self.hidden_size, self.hidden_size), prefix + "Wd_xh2")
            self.bd_xh2 = init_bias(self.hidden_size, prefix + "bd_xh2")
            self.Wd_hy = init_weights((self.hidden_size, self.out_size), prefix + "Wd_hy")
            self.bd_hy = init_bias(self.out_size, prefix + "bd_hy")
            self.params = [self.Wd_xh, self.bd_xh, self.Wd_xh2, self.bd_xh2, self.Wd_hy, self.bd_hy]
        
        def discriminate(self, x):
            h0 = T.tanh(T.dot(x, self.Wd_xh) + self.bd_xh)
            h1 = T.tanh(T.dot(h0, self.Wd_xh2) + self.bd_xh2)
            y = T.dot(h1, self.Wd_hy) + self.bd_hy
            return y

    def cost_nll(self, pred, label):
        cost = -T.log(pred) * label
        cost = T.mean(T.sum(cost, axis = 1))
        return cost

    def cost_mse(self, pred, label):
        cost = T.mean((pred - label) ** 2)
        return cost


    def define_train_test_funcs(self):
        G = self.Generator(self.in_size, self.latent_size, self.hidden_size)
        D = self.Discriminator(self.in_size,  self.hidden_size)
        self.params_dis = D.params
        self.params_gen = G.params

        g = G.generate(self.Z)
        d1 = D.discriminate(self.X)
        d2 = D.discriminate(g)

        loss_d = -T.mean(d1) + T.mean(d2)
        gparams_d = []
        for param in self.params_dis:
            gparam = T.grad(loss_d, param)
            gparams_d.append(gparam)

        nll = self.cost_nll(g, self.X)
        loss_g = -T.mean(d2) +  nll
        gparams_g = []
        for param in self.params_gen:
            gparam = T.grad(loss_g, param)
            gparams_g.append(gparam)

        lr = T.scalar("lr")
        optimizer = eval(self.optimizer)
        
        updates_d = optimizer(self.params_dis, gparams_d, lr)
        clip_updates_d = []
        for p, v in updates_d: 
            clip_updates_d.append((p, T.clip(v, -0.01, 0.01)))
        updates_d = clip_updates_d
        
        updates_g = optimizer(self.params_gen, gparams_g, lr)  

 
        self.train_d = theano.function(inputs = [self.X, self.Z, lr],
                outputs = loss_d, updates = updates_d)
        self.train_g = theano.function(inputs = [self.X, self.Z, lr],
                outputs = loss_g, updates = updates_g)
        
        self.generate = theano.function(inputs = [self.Z],
                outputs = g)


