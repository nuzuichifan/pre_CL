import pandas as pd
import math

def mae(action_predict, action_true):
    """
        计算MAE
    """
    N = len(action_true)  # 测试集样本数
    sum = 0  # 计算求和

    for i in range(N):  # 计算 |真实值-预测值|的总和
        sum = sum + abs(action_true[i] - action_predict[i])

    MAE = sum / N

    print("评价指标MAE的值为：", str(MAE))

def rmse(action_predict, action_true):
    """
        计算RMSE
    """
    sum = 0
    N = len(action_true)  # 测试集样本数

    for i in range(N):
        sum = sum + math.pow(action_true[i] - action_predict[i], 2)

    RMSE = math.sqrt(sum / N)

    print("评价指标RMSE的值为：", str(RMSE))

def cv(action_predict, action_true):
    """
        计算CV
    """
    sum = 0
    N = len(action_true)  # 测试集样本数

    for i in range(N):
        sum = sum + math.pow(action_true[i] - action_predict[i], 2)

    RMSE = math.sqrt(sum / N)

    sum = 0
    for i in range(N):
        sum = sum + action_true[i]

    actual_mean = sum / N  # 求真实值的平均值

    CV = RMSE / actual_mean

    print("评价指标CV的值为：", str(CV))

def r2(action_predict, action_true):
    """
        计算R^2
    """
    sum_molecule = 0
    sum_denominator = 0

    N = len(action_true)  # 测试集样本数
    sum = 0
    for i in range(N):
        sum = sum + action_true[i]
    actual_mean = sum / N  # 求真实值的平均值

    for i in range(N):
        sum_molecule = sum_molecule + math.pow(action_predict[i] - actual_mean, 2)  # 分子求和
        sum_denominator = sum_denominator + math.pow(action_true[i] - actual_mean, 2)  # 分母求和

    print("评价指标R^2的值：", str(sum_molecule / sum_denominator))

def xxx(predict,true):
    mae(predict, true)
    rmse(predict, true)
    cv(predict,true)
    r2(predict,true)
df=pd.read_excel("result.xls")
ture_data=df['ture']
BP_data=df['BP']
LSTM_data=df['LSTM']
DDPG_data=df['DDPG']
ddpg0_data=df['ddpg0']
ddpg1_data=df['ddpg1']
ddpg2_data=df['ddpg2']
ddpg3_data=df['ddpg3']
ddpg4_data=df['ddpg4']
ddpg5_data=df['ddpg5']
ddpga0_data=df['ddpga0']
ddpga1_data=df['ddpga1']
ddpga2_data=df['ddpga2']
ddpga3_data=df['ddpga3']
ddpga4_data=df['ddpga4']
ddpga5_data=df['ddpga5']
xxx(ddpg0_data.values,ture_data.values)
# xxx(ddpg1_data.values,ture_data.values)
# xxx(ddpg2_data.values,ture_data.values)
# xxx(ddpg3_data.values,ture_data.values)
# xxx(ddpg4_data.values,ture_data.values)
# xxx(ddpg5_data.values,ture_data.values)

xxx(ddpga0_data.values,ture_data.values)
xxx(ddpga1_data.values,ture_data.values)
xxx(ddpga2_data.values,ture_data.values)
xxx(ddpga3_data.values,ture_data.values)
xxx(ddpga4_data.values,ture_data.values)
xxx(ddpga5_data.values,ture_data.values)
#print(ture_data)