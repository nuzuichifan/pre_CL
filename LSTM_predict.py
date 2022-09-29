import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import time

# torch.manual_seed(202226)
# np.random.seed(202226)

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=30, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size * 30, 30)
        self.linear1 = nn.Linear(30, 1)



    def forward(self, input_seq):

        input_seq1 = torch.reshape(input_seq, (30, 1))
        # print(torch.unsqueeze(input_seq1, 1).shape)
        lstm_out, _ = self.lstm(torch.unsqueeze(input_seq1, 0))
        #lstm_out, _ = self.lstm(input_seq.view(len(input_seq), 1, -1))

        # print(lstm_out.view(1, -1).shape)
        predictions = self.linear(lstm_out.view(1, -1))
        predictions1 = torch.softmax(predictions, dim=1)
        # print(predictions1.shape)
        predictions = predictions * predictions1
        # print(predictions.shape)
        predictions = torch.sigmoid(self.linear1(predictions))
        # print(predictions.shape)
        return predictions

train_window = 16

Data = pd.read_csv("pre_02.csv")
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
model = LSTM()
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
true_test.to_csv("LSTM_data/lstm.csv")
loss.to_csv("LSTM_data/lstm_loss.csv")





