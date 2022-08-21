import numpy as np
from funcs import *

class TwoLayNet(object):
    
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.random.randn(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.random.randn(output_size)

        # 生成层
        self.layers = OrderedDict()
        
        self.layers['Affine1'] = Affine(self.params['W1'],self.params['b1'])
        self.layers['Relu1'] = ReLu()
        
        self.layers['Affine2'] = Affine(self.params['W2'],self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()
        
    def predict(self, x):
        # 到第二层神经网络输出为止，比如 a1=x*W1+b1 z1=relu(a1) x=z1*W2+b2，之后返回x，后面不进行softmax的处理 
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    # x:输入数据 t:监督数据
    def loss(self, x, t):
        y = self.predict(x)
        # 在 lastLayer 层中 进行 softmax(y) 之后接着求损失函数 cross_entropy_error(softmax(y),t)
        return self.lastLayer.forward(y,t)
    
    # x:输入数据 t:监督数据
    def accuracy(self, x, t):
        # 根据初始输入数据求精度
        y=self.predict(x) # y是经过两层神经网络的输出，不需要经过softmax处理（如果要求损失函数，则需要经过softmax的处理，将输出和监督数据统一）

        y = np.argmax(y, axis=1) #y=[[2,10,4],[3,9,1]] --> y=[1,1]
        if t.ndim != 1: # 如果 t 是 one-hot 的形式
            t = np.argmax(t, axis=1) # t=[[0,1,0],[0,1,0]] --> t=[1,1]

        accu = np.sum(y==t) / x.shape[0]
        
        return accu
            
    # x:输入数据 t:监督数据
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x,t)

        grads={}
        
        # grads['Wn/bn'] 的 shape 和 Wn/bn 的 shape 一致 
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads
    
    def gradient(self, x, t):
        # 要想实现反向传播-backward，得先把正向-forward 走完
        # 虽然 self.loss 已经走完了正向，但是 self.gradient 比 self.loss 要先调用
        self.loss(x,t)

        # 下面走 backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads



class MultiLayerNet(object):
    """全连接的多层神经网络
    Parameters
    ----------
    input_size : 输入大小（MNIST的情况下为784）
    hidden_size_list : 隐藏层的神经元数量的列表（e.g. [100, 100, 100]）
    output_size : 输出大小（MNIST的情况下为10）
    activation : 'relu' or 'sigmoid'
    weight_init_std : 指定权重的标准差（e.g. 0.01）
        指定'relu'或'he'的情况下设定“He的初始值”
        指定'sigmoid'或'xavier'的情况下设定“Xavier的初始值”
    weight_decay_lambda : Weight Decay（L2范数）的强度，初始值为0
    """
    def __init__(self, input_size, hidden_size_list, output_size, \
                 activation='relu', weight_init_std='he', weight_decays_lambda=0):
        
        self.input_size = input_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.output_size = output_size
        self.weight_decay_lambda = weight_decays_lambda

        # 初始化权重参数
        self.params = {}
        self.__init_weight(weight_init_std)

        # 创建层
        self.layers = OrderedDict()
        activation_layers = {'relu': ReLu, 'sigmoid': Sigmoid}

            # 创建除了输出层以为的层
        for idx in range(1, self.hidden_layer_num+1):
            # 传给 Affine 的 W 和 b 是地址，当 self.params['Wn/bn'] 修正之后，Affine 中的 W 和 b 也会改变
            self.layers['Affine' + str(idx)] = Affine(self.params['W'+str(idx)], self.params['b'+str(idx)])
            self.layers[activation.capitalize() + str(idx)] = activation_layers[activation.lower()]()

            # 创建输出层
        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W'+str(idx)], self.params['b'+str(idx)])

        self.last_layer = SoftmaxWithLoss()
        

    def __init_weight(self, weight_init_std):
        """设定权重的初始值

        Parameters
        ----------
        weight_init_std : 指定权重的标准差（e.g. 0.01）
            指定'relu'或'he'的情况下设定“He的初始值”
            指定'sigmoid'或'xavier'的情况下设定“Xavier的初始值”
        """
        all_layers_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, self.hidden_layer_num+2): # range(1, len(all_layers_size_list)) 也可以
            scale = weight_init_std
            if str(scale).lower() in ('relu','he'): # 使用 ReLu 情况下的初始值
                scale = np.sqrt(2.0 / all_layers_size_list[idx-1])
            elif str(scale).lower() in ('sigmoid','xavier'): # 使用 Sigmoid 情况下的初始值
                scale = np.sqrt(1.0 / all_layers_size_list[idx-1])

            self.params['W' + str(idx)] = scale * np.random.randn(all_layers_size_list[idx-1],all_layers_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_layers_size_list[idx])

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x
    
    def loss(self, x, t):
        """
        Parameters
        ----------
        x : 输入数据
        t : 教师标签

        Returns  
        ----------
        损失函数的值
        """
        y = self.predict(x)

        weight_decay = 0
        for idx in range(1, self.hidden_layer_num+2):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)

        loss = self.last_layer.forward(y, t) + weight_decay
        return loss
        
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        
        acc = np.sum(y==t) / float(x.shape[0])
        return acc
    
    def numerical_gradient(self, x, t):
        """
        数值微分求梯度
        Parameters
        ----------
        x : 输入数据
        t : 教师标签

        Returns
        ----------
        梯度grads - grads 是字典
            grads['W1']、grads['W2']、...是各层的权重
            grads['b1']、grads['b2']、...是各层的偏置
        """
        grads = {}
        loss_W = lambda W: self.loss(x,t)
        for idx in self.params.keys():
            grads[idx] = numerical_gradient(loss_W, self.params[idx])

        return grads
    
    def gradient(self, x, t):
        """
        数值微分求梯度
        Parameters
        ----------
        x : 输入数据
        t : 教师标签

        Returns
        ----------
        梯度grads - grads 是字典
            grads['W1']、grads['W2']、...是各层的权重
            grads['b1']、grads['b2']、...是各层的偏置
        """
        grads = {}
        self.loss(x, t)

        dout = 1
        dout = self.last_layer.backward(dout)

        layers = reversed(list(self.layers.values()))

        for layer in layers:
            dout = layer.backward(dout)

        for idx in range(1, self.hidden_layer_num+2):
            grads[f'W{idx}'] = self.layers[f'Affine{idx}'].dW + self.weight_decay_lambda * self.layers[f'Affine{idx}'].W
            grads[f'b{idx}'] = self.layers[f'Affine{idx}'].db

        return grads

        