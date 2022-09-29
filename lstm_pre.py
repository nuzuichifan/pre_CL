from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import time
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt


list_=[ '地面气压(hPa)', '气温2m(℃)', '地表温度(℃)', '露点温度(℃)', '相对湿度(%)', '北向风速(V,m/s)', '东向风速(U,m/s)',
        '总太阳辐射度(down,J/m2)', '净太阳辐射度(net,J/m2)', '紫外强度(J/m2)', 'year', 'mouth', 'day', 'hour',
       '冷负荷（kW）']
df=pd.read_excel("pre_data.xls",sheet_name='Sheet2')
# print([column for column in df])
 # 1. 实例化转换器（feature_range是归一化的范围，即最小值-最大值）
transfer = MinMaxScaler(feature_range=(0, 1))
# 2. 调用fit_transform （只需要处理特征）
df= transfer.fit_transform(df[list_])
df=pd.DataFrame(df)
#print(df)

x_train = df.iloc[0:5888,:14] #训练数据

y_train = df.iloc[0:5888,14] #训练数据目标值

x_train = x_train.values.reshape(-1, 1, 14) #将训练数据调整成pytorch中lstm算法的输入维度
y_train = y_train.values.reshape(-1, 1, 1)  #将目标值调整成pytorch中lstm算法的输出维度
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)
#print(x_train,y_train)
class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN,self).__init__() #面向对象中的继承
        self.lstm = torch.nn.LSTM(14,64,2) #输入数据14个特征维度，64个隐藏层维度，2个LSTM串联，第二个LSTM接收第一个的计算结果
        self.out = torch.nn.Linear(64,1) #线性拟合，接收数据的维度为64，输出数据的维度为1
    def forward(self,x):
        x1,_ = self.lstm(x)
        a,b,c = x1.shape
        out = self.out(x1.view(-1,c)) #因为线性层输入的是个二维数据，所以此处应该将lstm输出的三维数据x1调整成二维数据，最后的特征维度不能变
        out1 = out.view(a,b,-1) #因为是循环神经网络，最后的时候要把二维的out调整成三维数据，下一次循环使用
        return out1
rnn = RNN()
optimizer = torch.optim.Adam(rnn.parameters(),lr = 0.02)
loss_func = torch.nn.MSELoss()
for i in range(2000):
    var_x = Variable(x_train).type(torch.FloatTensor)
    var_y = Variable(y_train).type(torch.FloatTensor)
    out = rnn(var_x)
    loss = loss_func(out,var_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('Epoch:{}, Loss:{:.5f}'.format(i+1, loss.item()))

dataX1 = (df.iloc[5888:8832,:14]).values.reshape(-1, 1, 14)
dataX2 = torch.from_numpy(dataX1)
var_dataX = Variable(dataX2).type(torch.FloatTensor)

pred = rnn(var_dataX)

pred_test = pred.view(-1).data.numpy()

plt.plot(pred.view(-1).data.numpy(), 'r', label='prediction')
ture=df.iloc[5888:8832,14]
print(mean_squared_error(ture.reset_index(drop=True),pred.view(-1).data.numpy()))
# print(np.abs(df.iloc[5888:8832,14]-pred.view(-1).data.numpy()).mean())  # 模型评价
plt.plot((ture.reset_index(drop=True)), 'b', label='real')
plt.legend(loc='best')
plt.show()