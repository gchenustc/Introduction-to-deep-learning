# 下面函数用来调用
import numpy as np

# 激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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

    delta = 1e-7

    if y.ndim == 1:  # y = [1,2,4] t = [0,0,1]
        # 如果 ndim 为 1，为了统一，转换为 2 维
        y = y.reshape(1,y.size)  # [[1,2,4]]
        t = t.reshape(1,t.size)  # [[0,0,1]]

    if t.size == y.size:  # 将 one-hot 形式转换为普通形式 比如 t=[[0,0,1],[0,1,0]] -->  t=[2,1]
        t = t.argmax(axis=1)
    
    batch_size = y.shape[0] # 数据个数 
    # return -np.sum(t*np.log(y + delta)) / batch_size
    return -np.sum(np.log(y[np.arange(batch_size),t])) / batch_size


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
    """2d数组的梯度"""

    if X.ndim == 1:
        return _numerical_gradient_1d(f,X)
    else:
        grad = np.zeros_like(X) # X=[[2,3,4],[1,2,1]], grad=[[0,0,0],[0,0,0]]
        
        for idx, x in enumerate(X): #  x=[2,3,4],[1,2,1], idx=0,1
            grad[idx] = _numerical_gradient_1d(f,x)
        
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


# 两层神经网络的类
class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size) # randn生成的数组均值为0，方差为1，weight_init_std用来改变方差
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
    
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y
        
    # x:输入数据, t:监督数据
    def loss(self, x, t):
        y = self.predict(x)
        
        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x) # 假设 x=[[1,2,3],[3,2,1]], y=[[0.1,0.2,0.3,0.4],[0.5,0.3,0.1,0.1]], t=[[0,0,0,1],[0,1,0,0]]
        y = np.argmax(y, axis=1) # y = [3,0]
        t = np.argmax(t, axis=1) # t = [3,1]
        
        accuracy = np.sum(y == t) / float(x.shape[0]) # np.sum([3,0]==[3,1]) / float(2) = 0.5
        return accuracy
        
    # x:输入数据, t:监督数据
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient_2d(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient_2d(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient_2d(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient_2d(loss_W, self.params['b2'])
        
        return grads