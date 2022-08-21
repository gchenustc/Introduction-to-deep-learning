import numpy as np
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataset.mnist import load_mnist
from funcs import *

class Momentum(object):
    def __init__(self, momentum=0.9, lr=0.1):
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

class SGD(object):
    def __init__(self, lr):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]



### ------- 下面是测试 ------

# 读入训练数据
(x_train, t_train), (x_test,t_test) = load_mnist(normalize=True, one_hot_label=True)

# 建立神经网络实例
network = TwoLayNet(784, 50, 10)

# 一些参数的设置
train_size = x_train.shape[0]
batch_size = 100
iter_per_epoch = max(1, int(train_size/batch_size))

iters_num = 10000

learn_rate = 0.01  # 在更新参数的在SGD类中传入

train_loss_list = list()
train_acc_list = list()
test_acc_list = list()

# --------------  Momentum 类的创建 ---------------
optimizer = Momentum(momentum=0.1, lr=learn_rate)
# optimizer = SGD(lr=learn_rate)

# 迭代开始
for i in range(iters_num):

    # 取出小批量训练数据，这里是100个
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 求梯度
    grads = network.gradient(x_batch, t_batch)

    # 更新梯度
    params = network.params
    optimizer.update(params, grads)

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 判断是否达到一次 epoch
    if not (i+1)%iter_per_epoch:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print('第%d次训练...' % (i+1))
        print('loss_function: %.8f. train_accuracy: %8f. test_accuracy: %.8f' % (loss, train_acc, test_acc))