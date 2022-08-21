from pickletools import optimize
import sys, os
current_dir = os.path.dirname(__file__)
last_dir = os.path.dirname(current_dir)
sys.path.append(last_dir)  # 函数所在位置
sys.path.append(os.path.dirname(last_dir))  # mnist所在位置

import numpy as np
import matplotlib.pyplot as plt
from optimizer import *


def f(x,y):
    return (1/20) * x**2 + y**2


def df(x,y):
    return (1/10)*x, 2*y


# 设置初始位置
init_pos = [-7.0, 2.0]

# 定义循环次数
iters_num = 17

# 定义 optimizer 字典
optimizer = {}
optimizer['SGD'] = SGD(0.95)
optimizer['Momentum'] = Momentum(0.1)
optimizer['AdaGrad'] = AdaGrad(1)
optimizer['Adam'] = Adam(0.5)

# 定义测试第 n 个 optimizer 
iter = 1

# 逐个测试每个 optimizer
for key in optimizer:
    # 初始化 params 和 grads
    params={}
    grads={}
    params['x'] = init_pos[0]
    params['y'] = init_pos[1]

    # 记录x，y轴的变化，用于绘图
    x_history_list = []
    y_history_list = []

    # 计算每个 optimizer 的性能，梯度下降 iters_num 次
    for i in range(iters_num):
        x_history_list.append(params['x'])
        y_history_list.append(params['y'])

        grads['x'], grads['y'] = df(params['x'],params['y'])
        optimizer[key].update(params, grads)

    # 绘图主体
    x_contour = np.arange(-10,10,0.01)
    y_contour = np.arange(-5,5,0.01)
    x_contour, y_contour = np.meshgrid(x_contour, y_contour)
    z_contour = f(x_contour, y_contour)

    # simplize contoure
    mask = z_contour > 7
    z_contour[mask] = 0

    plt.subplot(2,2,iter)
    iter += 1
    plt.contour(x_contour, y_contour, z_contour)
    plt.plot(x_history_list,y_history_list,'ro-') # 红色，同时包含圆圈和线条
    
    # 绘图细节
    plt.xlabel('x')
    plt.xlabel('y')
    plt.title(key)

plt.show()

