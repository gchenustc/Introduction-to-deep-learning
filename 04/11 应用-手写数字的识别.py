import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataset.mnist import load_mnist
from funcs import *
import sys, os
import matplotlib.pyplot as plt


# x_train, x_test 元素为0-1的数字, 维度 n*784
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

net = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

train_size = x_train.shape[0] # 60000
batch_size = 100 # 每次循环取出的数据个数(图片的个数)
iters_num = 10000 # 设定循环的次数，每次取 batch_size 个数据
learn_rate = 0.1 # 权重 -= 学习率*梯度

iter_per_epoch = max(int(train_size / batch_size), 1) # 每次每次取的batch_size个数据都不重复，则需要iter_per_epoch次取完

train_loss_list = list()
train_acc_list = list()
test_acc_list = list()

for i in range(iters_num):
    # 从所有样本中(60000) 取 batch_size(100) 个数据
    batch_mask = np.random.choice(train_size, batch_size) # [59,1,2..39(一共100个数据)]
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 梯度 - 损失函数对权重系数的梯度
    grad = net.numerical_gradient(x_batch,t_batch)
    
    # 更新参数
    for key in ('W1','b1','W2','b2'):
        net.params[key] -= learn_rate*grad[key]
        
        loss = net.loss(x_batch, t_batch)
        train_loss_list.append(loss)

    print(f'第{i}次训练...')
    if i % iter_per_epoch == 0:
        train_acc = net.accuracy(x_batch, t_batch)
        test_acc = net.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# 绘制图形
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
        