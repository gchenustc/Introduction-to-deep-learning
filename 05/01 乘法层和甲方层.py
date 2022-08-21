class MulLayer(object):
    def __init__(self):
        self.x = None
        self.y = None
        
    def forward(self,x,y):
        self.x = x
        self.y = y
        return self.x * self.y
    
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy


class AddLayer(object):
    def __init__(self):
        pass
    
    def forward(self,x,y):
        out = x + y
        return out
    
    def backward(self, dout):
        dx = 1 * dout
        dy = 1 * dout
        return dx, dy
    

## 购买两个苹果，每个100元，买三个橘子，每个橛子150元，消费税为1.1
apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_origin_layer = AddLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple_num, apple)
orange_price = mul_orange_layer.forward(orange_num, orange)
all_price = add_apple_origin_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(all_price, tax)

print(price)

# backward
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_apple_origin_layer.backward(dall_price)
dapple_num, dapple  = mul_apple_layer.backward(dapple_price)

print(dapple)