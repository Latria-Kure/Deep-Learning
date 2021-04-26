from source.mnist import load_mnist
from source.Network import TwoLayersClassifyNetwork
import numpy as np
# data preparation
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# model fitting
model=TwoLayersClassifyNetwork(hidden_size=60)
# hidden_size depends on user
model.fit(x_train,t_train,iter_num=10000,batch_size=100)

# model predicting
prediction=model.predict(x_test)
predict_result=prediction@np.arange(10).reshape(-1,1).reshape(-1)
correct_result=t_test@np.arange(10).reshape(-1,1).reshape(-1)

# accuracy
accuracy=np.sum(predict_result==correct_result)/t_test.shape[0]
print(accuracy)