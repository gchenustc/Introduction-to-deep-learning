import numpy as np
import matplotlib.pyplot as plt
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # 为了导入父目录的文件而进行的设定
from util import smooth_curve
from funcs import *
from optimizer import *
from networks import *
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))  # 为了导入父目录的文件而进行的设定
from dataset.mnist import load_mnist


# 0. 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True) # 注意t没有 one-hot
train_size = x_train.shape[0]
batch_size =128
iter_per_print = 100 # 每 100 次循环打印一次信息
iters_num = 3000 # 循环次数

# 1. 创建每个 optimizer 的 神经网络，以及 损失值，训练精确度，测试用于后面画图
optimizer_dict = {'SGD': SGD, 'Momentum': Momentum, 'AdaGrad': AdaGrad, 'Adam': Adam}
optimizers = {}
networks = {}
train_loss = {}
train_accs = {}
test_accs = {}
for key in optimizer_dict:
    optimizers[key] = optimizer_dict[key]()
    networks[key] = MultiLayerNet(784, [100,100,100,100], 10)
    train_loss[key] = []
    train_accs[key] = []
    test_accs[key] = []


# 2. 训练开始
for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    for key in optimizers:
        loss = networks[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)

        grads = networks[key].gradient(x_batch, t_batch)
        optimizers[key].update(networks[key].params, grads)

    # 每 iter_print 次，打印一次loss的信息，并计算精确度
    if not i%iter_per_print:
        print(f"======= current is the {i}'s circle... =======")

        for key in optimizers:
            train_acc = networks[key].accuracy(x_train, t_train)
            test_acc = networks[key].accuracy(x_test, t_test)
            train_accs[key].append(train_acc)
            test_accs[key].append(test_acc)
            print(f'train_acc for {key}: {train_acc}; test_acc for {key}: {test_acc}')


        for key in optimizers:
            print(f'loss for {key}: {train_loss[key][-1]}')
        

# 3. 绘图
markers = {"SGD": "o", "Momentum": "x", "AdaGrad": "s", "Adam": "D"}
plt.figure(figsize=(25,10))

# loss
plt.subplot(1,3,1)
x = np.arange(iters_num)
for key in optimizers:
    plt.plot(x, smooth_curve(train_loss[key]), markers[key], label=key)
plt.ylim(0, 3)
plt.title('compared with loss')
plt.xlabel('step')
plt.ylabel('loss')
plt.legend(loc = "upper right")

# train_acc and test_acc
markers = {"SGD": "o-", "Momentum": "x-", "AdaGrad": "s-", "Adam": "D-"}
# train_acc
plt.subplot(1,3,2)
x = np.linspace(0, iters_num, int(iters_num/iter_per_print))
for key in optimizers:
    plt.plot(x, train_accs[key], markers[key], label=key)
plt.title('compared with train_set accuracy')
plt.xlabel('step')
plt.ylabel('train_set accuracy')
plt.legend()

# test_acc
plt.subplot(1,3,3)
for key in optimizers:
    plt.plot(x, test_accs[key], markers[key], label=key)
plt.title('compared with test_set accuracy')
plt.xlabel('step')
plt.ylabel('test_set accuracy')
plt.legend()

plt.show()

