# 下面函数用来调用
import numpy as np
from collections import OrderedDict

# Affine
class Affine(object):
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        self.dW = None
        self.db = None

        self.original_x_shape = None
        # self.dx = None # 可以不设置，因为dx是直接返回的
    
    def forward(self, x): # x 传入的是二维数组(多个数据,或者单个数据reshape成二维)
        self.original_x_shape = x.shape # 如果传入的 x.shape 不是 [N,784] 的形式，而是 [N,1,28,28]的形式，先把shape备份下来再转换为[N,784]，之后反向传播输出dx时以 [N,1,28,28] 的形式输出
        x = x.reshape(x.shape[0],-1) # 如果 x.shape=[N,784]，则不变

        # 运行forward，输入的x会被类保留下来，后面反向传播求 dw=x.t*dout 用的到
        self.x = x  # 保留 self.x 用于 backward 计算 dw=x.T*dout
        
        out = np.dot(self.x, self.W) + self.b
        return out
    
    def backward(self, dout):
        # 虽然返回的是dx,但是dw和db会被类保留,dx=dout*W.T 中的W初始化就得到了，dw=x.T*dout 中的x在forward中传入
        # dW 和 db 是损失函数的梯度，也就最终需要的值
        # dW（db）和 W（b）的 shape 一致
        self.dW = np.dot(self.x.T,dout) 
        self.db = np.sum(dout,axis=0)
        
        dx = np.dot(dout,self.W.T)
        # dx.shape 从 [N,784] 的形式还原成 [N,1,28,28]的形式
        dx = dx.reshape(self.original_x_shape)
        return dx


#激活函数
class Sigmoid(object):
    def __init__(self):
        self.out = None
    
    def forward(self,x):
        self.out = 1/(1+np.exp(-x))
        return self.out

    def backward(self,dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx 


class ReLu(object):
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


def softmax(x):
    """输入一维数组或者二维数组，输入一维返回一维，输入二维返回二维"""
    if x.ndim == 2: # x=[[1,2,3],[2,3,4]]
        x = x.T # x=[[1,2],[2,3],[3,4]]
        x = x - np.max(x,axis=0) # x=x - [3,4] = [[1-3,2-4],[2-3,3-4],[3-3,4-4]] = [[-2,-2],[-1,-1],[0,0]]
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T # 返回二维数组
    # x.ndim == 1 的情况
    x = x-np.max(x)
    return np.exp(x) / np.sum(np.exp(x)) # 返回一维数组比如,ret.shape=(3,)


# 损失函数
def cross_entropy_error(y,t):
    
    """
    可以传入一维(单个数据)或者二维(多个数据)的 y (预测数据)，一维的会在函数内转换为二维
    传入的 t(监督数据) 可以是one-hot的形式，比如 t=[[0,0,1],[0,1,0]]，或者 t=[2,1]
    """

    if y.ndim == 1:  # y = [1,2,4] t = [0,0,1]
        # 如果 ndim 为 1，为了统一，转换为 2 维
        y = y.reshape(1,y.size)  # [[1,2,4]]
        t = t.reshape(1,t.size)  # [[0,0,1]]

    if t.size == y.size:  # 将 one-hot 形式转换为普通形式 比如 t=[[0,0,1],[0,1,0]] -->  t=[2,1]
        t = t.argmax(axis=1)
    
    batch_size = y.shape[0] # 数据个数 
    return -np.sum(np.log(y[np.arange(batch_size),t] + 1e-7)) / batch_size


class SoftmaxWithLoss(object):
    # 需要调用 激活函数softmax() 和 损失函数 cross_entropy_error()
    def __init__(self):
        self.y = None
        self.t = None
    
    def forward(self,x,t):
        # 传入的x是Affinx的forward函数的返回值，也就是[输入数据 * W + b]
        # self.y 和 self.t 在 forward 中传入后保存，然后在backword中被使用
        self.y = softmax(x) # 激活函数处理
        self.t = t

        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self,dout):
        # dout 其实为1，因为要求的是 Loss 对权重参数的梯度，Loss是终点，@loss/@loss = 1 (@这里指的是偏导)
        batch_size = self.t.shape[0]

        if self.y.size == self.t.size:
            # 如果是 one-hot 的形式
            dx = (self.y - self.t)/batch_size # 这里不懂为啥除以 batch_size，虽然梯度时原来的1/batch_size倍，但是梯度的方向是不变的
        else:
            # 非 one-hot 的形式
            dx = self.y.copy()
            dx[np.arange(batch_size),self.t] -= 1
            dx /= batch_size

        return dx


# 梯度
# 下面两个函数定义了多个数据的梯度，其中每个数据可以是一元或者多元函数
def _numerical_gradient_1d(f,x):
    """
    求一元或者多元函数梯度，传入的 x 是一维数组，代表坐标，浮点数。比如 x = [1.0,2.0] 就是二元函数在 (1,2) 上的点。求的是在这个点上的二元函数的两个方向的偏导
    """
    h = 1e-4

    grad = np.zeros_like(x) # 假如是二元函数，传入变量 x = [3,4]，则现在 grad = [0,0]，grad[0],grad[1] 分别是二元函数的两个变量的梯度

    for idx in range(x.size): # x: [3,4], idx: [0,1]
        tmp_val = x[idx] # tmp_val=x[0]=3
        x[idx] = tmp_val + h # x: [3+h,4]
        fxh1 = f(x) # [3+h,4] 对应的函数值 f
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1-fxh2) / (2*h)
        
        x[idx] = tmp_val # 还原x
        
    return grad


def numerical_gradient_2d(f,X):
    """多维数组(向下兼容)求梯度"""

    if X.ndim == 1:
        return _numerical_gradient_1d(f,X)
    else:
        grad = np.zeros_like(X) # X=[[2,3,4],[1,2,1]], grad=[[0,0,0],[0,0,0]]
        
        for idx, x in enumerate(X): #  x=[2,3,4],[1,2,1], idx=0,1
            grad[idx] = _numerical_gradient_1d(f,x)
        
        return grad

# 综合 numerical_gradient_2d/1d
def numerical_gradient(f,X):
    """多维数组(向下兼容)求梯度"""
    h = 1e-4
    grad = np.zeros_like(X)

    it = np.nditer(grad, flags=['multi_index'],op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index # 拿出索引，比如 (0,0) (0,1) (0,2) ... (1,0)
        tmp_var = X[idx]
        X[idx] = tmp_var + h
        fxh1 = f(X) # f(X_ori + h)
        X[idx] = tmp_var - h
        fxh2 = f(X) # f(X_ori - h)
        grad[idx] = (fxh1 - fxh2) / (2*h)

        X[idx] = tmp_var # 恢复 X
        it.iternext()

    return grad


# 梯度下降函数
def gradient_descent(f, init_x, lr=0.01, step_num=300):
    x = init_x # 假设是二元函数，x=[2,2], f=x[0]**2+x[1]**2
    x_history = []

    for i in range(step_num):
        x_history.append(x.copy())
        
        grad = numerical_gradient_2d(f,x)  # grad=[4,4]
        x -= lr * grad  # x = [2,2] - 0.01*[4,4] = [1.96,1.96]

    return x, np.array(x_history)