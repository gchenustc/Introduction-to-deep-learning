import os,sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataset.mnist import load_mnist
from funcs import *

# 读入数据
(x_train, t_train), (x_test,t_test) = load_mnist(normalize=True, one_hot_label=True)

network =  TwoLayNet(input_size=784, hidden_size=50, output_size=10)

train_size = x_train.shape[0]
batch_size = 100

iters_num = 10000
iter_per_epoch = max(1,int(train_size/batch_size))

learn_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.gradient(x_batch, t_batch)

    for key in ('W1','b1','W2','b2'):
        network.params[key] -= learn_rate * grad[key]
        
    loss = network.loss(x_batch, t_batch) # 这里计算损失函数最好不要用 x_train 和 t_train，因为会很慢

    train_loss_list.append(loss)

    if not (i+1)%iter_per_epoch:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print('第%d次训练...' % (i+1))
        print('loss_function: %.8f. train_accuracy: %8f. test_accuracy: %.8f' % (loss, train_acc, test_acc))