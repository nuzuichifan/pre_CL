import torch
import numpy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import time

np.set_printoptions(threshold=np.inf)
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
total_data_row=11776
total_data_col=15
# torch.manual_seed(202226)
# np.random.seed(202226)

class BP(nn.Module):
    def __init__(self, input_size=30, hidden_layer_size=100, output_size=1):
        super().__init__()

        #self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(input_size , hidden_layer_size)
        self.linear1 = nn.Linear(hidden_layer_size, output_size)



    def forward(self, input_seq):

        out0=self.linear(input_seq)
        out1=torch.softmax(out0,dim=1)
        out2=self.linear1(out1)
        out3=torch.sigmoid(out2)
        return out3

train_window = 16

Data = pd.read_csv("pre_02.csv")
Data = np.array(Data)
dataset = np.array(Data[0:total_data_row, 0:total_data_col])
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
        train_seq1 = input_data[tw + i:tw+1+i, 0:total_data_col-1]
        train_seq2= input_data[i: tw + i, total_data_col-1:total_data_col].T
        # print(train_seq1.shape)
        # print(train_seq2.shape)
        # train_seq = torch.cat((s1, s2))
        # print(train_seq.shape)
        train_seq = torch.cat((train_seq1, train_seq2),dim=1)
        train_label = input_data[i + tw : i + tw + 1, total_data_col-1:total_data_col]
        inout_seq.append((train_seq ,train_label))
    return inout_seq



train_inout_seq = create_inout_sequences(train_data_normalized, train_window)
model = BP()
model.to(device)


loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 100
LOSS = []
start = time.time()
for i in range(epochs):
    LOSSS = 0
    for seq, labels in train_inout_seq:
        # model.hidden_cell = (torch.zeros(1, 13, model.hidden_layer_size).to(device),
        #                 torch.zeros(1, 13, model.hidden_layer_size).to(device))
        y_pred = model(seq)
        single_loss = loss_function(y_pred, labels)
        optimizer.zero_grad()
        single_loss.backward()
        LOSSS += single_loss.item()
        optimizer.step()
    LOSS.append(LOSSS)
    if i % 1 == 0:
        print("epoch:{}, loss:{}".format(i, LOSSS))

end = time.time()
print(end - start)
model.eval()

test_energy = []
test_data_normalized = scaler1.fit_transform(test)
print(test_data_normalized.shape)
test_output = test_data_normalized
print(test_output.shape)
test_data_normalized = torch.FloatTensor(test_data_normalized).to(device)
print(test_data_normalized.shape)
test_inout_seq = create_inout_sequences(test_data_normalized, train_window)
test_output1 = test[:, total_data_col-1:total_data_col]

test_output2 = test_output[:, total_data_col-1:total_data_col]
a = np.max(test_output1)
b = np.min(test_output1)

# test_output1 = scaler2.fit_transform(test_output1)

test_output12 = []
i = 0
for seq, target in test_inout_seq:
    # s1 = test_data_normalized[train_window + i + 1, :-1]
    # s2 = test_data_normalized[i + 1: train_window + i + 1, -1].T
    # seq = torch.cat((s1, s2))

    test_output2[i + train_window, :] = model(seq).cpu().detach().numpy()
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
loss = pd.DataFrame(LOSS)
true_test = pd.DataFrame(test_output2)
true_test.to_csv("BP_data/BP.csv")
loss.to_csv("BP_data/BP_loss.csv")





