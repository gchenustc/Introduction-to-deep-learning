import numpy as np


class SGD(object):
    def __init__(self, lr=0.05):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class Momentum(object):
    def __init__(self, lr=0.005, momentum=0.9):
        self.momentum = momentum
        self.lr = lr
        self.v = None

    def update(self, params, grads):
        if self.v == None:
            self.v = dict()
            for key, value in params.items():
                self.v[key] = np.zeros_like(value)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


class AdaGrad(object):
    def __init__(self, lr=0.1):
        self.lr = lr
        self.h = None 
    
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= ( self.lr * grads[key] ) / (np.sqrt(self.h[key]) + 1e-7) # 1e-7是为了防止溢出


class Adam(object):
    
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999):
        self.theta = 1e-7

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

        self.iter = 0 # 用来记录 更新梯度的 次数

        self.v = None # 速度 类似 Momentum
        self.s = None # 用来在接近终点的时候减小学习率，类似 Adagrad

    def update(self, params, grads):
        if self.v is None:
            self.v, self.s = {}, {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
                self.s[key] = np.zeros_like(val)
        
        self.iter += 1.0  # 每调用一次，记录一下梯度更新的次数

        # 偏置修正
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        # 更新梯度
        for key in params.keys():
            # 更新 v 和 s
            # self.v[key] += (1 - self.beta1) * (grads[key] - self.v[key])
            # self.s[key] += (1 - self.beta2) * (grads[key]**2 - self.s[key])

            # 另一种写法 (更容易理解)
            self.v[key] = self.beta1*self.v[key] + (1-self.beta1)*grads[key]
            self.s[key] = self.beta2*self.s[key] + (1-self.beta2)*(grads[key]**2)

            grads_new = self.v[key] / (np.sqrt(self.s[key]) + self.theta)

            # 更新参数
            params[key] -= lr_t * grads_new