import numpy as np
from source.Layer import Affine,ReLU,SoftmaxWithLoss

class TwoLayersClassifyNetwork():
    def __init__(self,hidden_size):
        self.__hidden_size=hidden_size

    def fit(self,data,label,iter_num=10000,batch_size=100):
        self.__input_size=data.shape[1]
        self.__out_size=label.shape[1]
        Affine_1=Affine(self.__input_size,self.__hidden_size)
        Affine_2=Affine(self.__hidden_size,self.__out_size)
        ReLU_1=ReLU()
        Forward_Layers=[Affine_1,ReLU_1,Affine_2]
        Backward_Layers=Forward_Layers[::-1]
        Last_Layer=SoftmaxWithLoss()

        train_size = data.shape[0]

        for i in range(iter_num):
            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = data[batch_mask]
            t_batch = label[batch_mask]
            for i in Forward_Layers:
                x_batch=i.forward(x_batch)
            Last_Layer.forward(x_batch)
            dout=1
            dout=Last_Layer.backward(dout,t_batch)
            for i in Backward_Layers:
                dout=i.backward(dout)
        self.__Traning_Layers=Forward_Layers
        self.__Out_Layer=Last_Layer

    def predict(self,data):
        x=data
        for i in self.__Traning_Layers:
            x=i.forward(x)
        result=self.__Out_Layer.forward(x)
        return result

