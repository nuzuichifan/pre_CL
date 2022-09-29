from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
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
print(df)


model = MLPRegressor(hidden_layer_sizes=(10,), random_state=10,learning_rate_init=0.1)  # BP神经网络回归模型
model.fit(df.iloc[0:5888,:14],df.iloc[0:5888,14])  # 训练模型
pre = model.predict(df.iloc[5888:8832,:14])  # 模型预测
print(np.abs(df.iloc[5888:8832,14]-pre).mean())  # 模型评价
import matplotlib.pyplot as plt
plt.plot(pre)
ture=df.iloc[5888:8832,14]
plt.plot(ture.reset_index(drop=True))
plt.show()


