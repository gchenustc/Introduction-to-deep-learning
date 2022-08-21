import os,sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataset.mnist import load_mnist
from optimizer import SGD
from funcs import *
from networks import MultiLayerNet


# 1. 导入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize = True)

    # 减少训练集，再现过拟合现象
x_train = x_train[:500]
t_train = t_train[:500]

# 2. 设置参数
    # 设置权值衰减
weight_decay_lambad = 0.1

max_epochs = 40 # 最大训练的epoch - 训练次数 = max_epoch * int(train_size/batch_size)
epoch_cnt = 0 # 当前的 epoch
train_size = x_train.shape[0]
batch_size = 100
iter_per_epoch = max(1, int(train_size/batch_size))

optimizer = SGD()

# 3. 设置两个神经网络，一格有权值衰减，一个没有，用于对比
network_name_dict = {'have_weight_decay': weight_decay_lambad, 'no_weight_decay': 0}
networks = {}
train_loss = {}
train_accs = {}
test_accs = {}
for key in network_name_dict:
    networks[key] = MultiLayerNet(784, [100]*6, 10, weight_decays_lambda=network_name_dict[key])
    train_accs[key] = []
    test_accs[key] = []
    train_loss[key] = []

# 4. 训练开始
for i in range(10000000):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    for key in networks:
        grads = networks[key].gradient(x_batch, t_batch)
        optimizer.update(networks[key].params, grads)

        loss = networks[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)
    
    if not i%iter_per_epoch:
        for key in networks:
            train_acc = networks[key].accuracy(x_train, t_train)
            test_acc = networks[key].accuracy(x_test, t_test)

            train_accs[key].append(train_acc)
            test_accs[key].append(test_acc)

            
        print("epoch:" + str(epoch_cnt))
        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break

# 5. 绘图
markers = {'have_weight_decay': {'train':'ro-', 'test':'bs-'}, 'no_weight_decay': {'train':'rx-', 'test':'bD-'}}
x = np.arange(max_epochs)
for key in networks:
    plt.plot(x, train_accs[key], markers[key]['train'], label=key+'_train', markevery=10)
    plt.plot(x, test_accs[key], markers[key]['test'], label=key+'_test', markevery=10)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
plt.show()
            
        