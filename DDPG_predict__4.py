import Agent
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import time

import xlwt
workbook = xlwt.Workbook(encoding = 'utf-8')
worksheet_action = workbook.add_sheet('a')
worksheet_reward = workbook.add_sheet('reward')

train_window = 16

Data = pd.read_csv("pre_02.csv",encoding = 'utf-8')
Data = np.array(Data)
dataset = np.array(Data[0:11776, 0:15])
# print(dataset.shape)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size: len(dataset), :]

scaler = MinMaxScaler(feature_range=(0, 1))
scaler1 = MinMaxScaler(feature_range=(0, 1))
train_data_normalized = scaler.fit_transform(train)
train_data_normalized = torch.FloatTensor(train_data_normalized).to(device)

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw - 1):
        train_seq1 = input_data[tw + i:tw+1+i, 0:14]
        train_seq2= input_data[i: tw + i, 14:15].T
        # print(train_seq1.shape)
        # print(train_seq2.shape)
        # train_seq = torch.cat((s1, s2))
        # print(train_seq.shape)
        train_seq = torch.cat((train_seq1, train_seq2),dim=1)
        train_label = input_data[i + tw : i + tw + 1, 14:15]
        inout_seq.append((train_seq ,train_label))
    return inout_seq
train_inout_seq = create_inout_sequences(train_data_normalized, train_window)
#print(len(train_inout_seq))

s_dim=30
a_dim=1
a_high_bound=1.2
a_low_bound=-0.1
BATCH_SIZE=32
MEMORY_SIZE=8000
EPISODES=20
EP_STEPS=len(train_inout_seq)
print(EP_STEPS)
ddpg=Agent.DDPG_agent1(state_counters=s_dim, action_counters=a_dim, batch_size=BATCH_SIZE, memory_size=MEMORY_SIZE,)
index_=0
var = 3  # the controller of exploration which will decay during training process
t1 = time.time()
for i in range(EPISODES):

#    s=train_inout_seq[0:1,:]
    ep_r = 0
    index_=0

    # for seq, labels in train_inout_seq:
    for j in range(EP_STEPS):
        # print(train_inout_seq[j])
        s,lab=train_inout_seq[j]
        if j==9402:
            s_ = torch.rand(1,30)
        else:
            s_,lab_=train_inout_seq[j+1]
            #print(s_.shape)
        # add explorative noise to action
        a = ddpg.choose_action(s)
        a = np.clip(a, a_low_bound, a_high_bound)
        #print(float(a),float(lab))
        r=-80*abs(float(a)-float(lab))

       # print(float(a), float(lab),r)
        #print(s.squeeze(), a.squeeze(), r , s_.squeeze())
        ddpg.store_memory(s.squeeze(), a.squeeze(), r , s_.squeeze())  # store the transition to memory
        worksheet_action.write(index_,i , float(a))
        if ddpg.index_memory > ddpg.memory_size:
            var *= 0.9995  # decay the exploration controller factor
            ddpg.learn()

        ep_r += r
        index_=index_+1
    print('Episode: ', i,'ep_r: ' ,ep_r)
    worksheet_reward.write(i, 0, ep_r)
print('Running time: ', time.time() - t1)
workbook.save('DDPG_data/DDPG_predict__4.xls')


actnet=ddpg.Actor_Net_eval
actnet.eval()

test_energy = []
test_data_normalized = scaler1.fit_transform(test)
print(test_data_normalized.shape)
test_output = test_data_normalized
print(test_output.shape)
test_data_normalized = torch.FloatTensor(test_data_normalized).to(device)
print(test_data_normalized.shape)
test_inout_seq = create_inout_sequences(test_data_normalized, train_window)
test_output1 = test[:, 14:15]

test_output2 = test_output[:, 14:15]
a = np.max(test_output1)
b = np.min(test_output1)

# test_output1 = scaler2.fit_transform(test_output1)

test_output12 = []
i = 0
for seq, target in test_inout_seq:
    # s1 = test_data_normalized[train_window + i + 1, :-1]
    # s2 = test_data_normalized[i + 1: train_window + i + 1, -1].T
    # seq = torch.cat((s1, s2))

    test_output2[i + train_window, :] = actnet(seq).cpu().detach().numpy()
    i += 1
    # test_output12.append(model(seq).item())
    # print(test_output1[i + train_window:i + train_window+1, -1])
    # print(model(seq).item())

test_output2[:, 0:1] = (a - b) * np.array(test_output2[:, 0:1]) + \
                b

# print(np.max(test_output1))
# actual_predictions = scaler2.inverse_transform(test_output1)
# print(actual_predictions.shape)
# actual_predictions = pd.DataFrame(test_output1)
# actual_predictions.to_csv("test_output_cnn.csv")
#loss = pd.DataFrame(LOSS)
true_test = pd.DataFrame(test_output2)
true_test.to_csv("DDPG_data/ddpg__4.csv")
#loss.to_csv("37_loss_3_att.csv")
