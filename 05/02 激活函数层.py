# ReLu(Rectified Linear Unit)激活函数
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
        return dout


class Sigmoid(object):
    def __init__(self):
        self.out = None
    
    def forward(self,x):
        self.out = 1/(1+np.exp(-x))
        return self.out

    def backward(self,dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx 


import numpy as np
relu_layer = ReLu()
relu_x = np.array([-2,3,4])
relu_out = relu_layer.forward(relu_x)
drelu_x = relu_layer.backward(np.array([1,1,1]))
print(relu_out)
print(drelu_x)
print('--- ---')

sigmoid_x = np.array([-2,-1,0,1,2])
sigmoid_layer = Sigmoid()
y = sigmoid_layer.forward(sigmoid_x)
dx = sigmoid_layer.backward(1)
print(y,dx)