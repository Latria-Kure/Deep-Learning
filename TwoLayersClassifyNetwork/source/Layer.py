import scipy.stats as st
import numpy as np
class MulLayer():
    def __init__(self):
        self.x=None
        self.y=None
        
    def forward(self,x,y):
        self.x=x
        self.y=y
        out=x*y
        return out
    
    def backward(self,dout):
        dx=dout*self.y
        dy=dout*self.x
        return dx,dy
    
class AddLayer():
    def __init__(self):
        self.x=None
        self.y=None
        
    def forward(self,x,y):
        self.x=x
        self.y=y
        out=x+y
        return out
    
    def backward(self,dout):
        dx=dout
        dy=dout
        return dx,dy

class ReLU():
    def __init__(self):
        self.input=None

    def forward(self,input):
        self.input=input
        GE_zero=input>0
        GE_zero.astype('int64')
        self.deriv=GE_zero
        return input*self.deriv

    def backward(self,dout):
        return dout*self.deriv

class Affine():
    def __init__(self,input_size,out_size):
        self.W=np.random.randn(input_size,out_size)*0.01
        self.B=np.zeros(out_size)

    def forward(self,input):
        self.input=input
        return self.input@self.W+self.B

    def backward(self,dout):
        lr=0.1
        self.dinput=dout@self.W.T
        dW=self.input.T@dout
        dB=np.sum(dout,0)
        self.W=self.W-lr*dW
        self.B=self.B-lr*dB
        return self.dinput

class SoftmaxWithLoss():
    def __init__(self):
        self.input=None

    def forward(self,input):
        self.input=input
        c=np.max(input,1).reshape(-1,1)
        softmax=np.exp(input-c)/np.sum(np.exp(input-c),1).reshape(-1,1)
        self.softmax=softmax
        result=softmax==np.max(softmax,1).reshape(-1,1)
        result=result.astype('int64')
        # predict result with ont-hot display
        return result

    def backward(self,dout,label):
        self.label=label
        batch_size=self.softmax.shape[0]
        dinput=self.softmax-self.label
        self.dinput=dinput/batch_size
        return self.dinput
