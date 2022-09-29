import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

# class Actor_Net_att(nn.Module):
#
#     def __init__(self, state_dim, action_dim):
#         super(Actor_Net_att, self).__init__()
#         self.fc0 = nn.Linear(state_dim, state_dim)
#         self.fc0.weight.data.normal_(0, 0.1)
#         self.fc0.bias.data.normal_(0.1)
#         self.fc1 = nn.Linear(state_dim, 32)
#         self.fc1.weight.data.normal_(0, 0.1)
#         self.fc1.bias.data.normal_(0.1)
#         self.fc3 = nn.Linear(32, action_dim)
#         self.fc3.weight.data.normal_(0, 0.1)
#         self.fc3.bias.data.normal_(0.1)
#
#     def forward(self, x):
#         out0 = F.softmax(self.fc0(x), dim=1)
#         out_ = x*out0
#         print("--")
#         print(x.shape)
#         print(out0.shape)
#         print(out_.shape)
#         print("--")
#         out1 = torch.relu(self.fc1(out_))
#         out2 = torch.sigmoid(self.fc3(out1))
#         return out2

class Actor_Net_att(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Actor_Net_att, self).__init__()
        self.fc1 = nn.Linear(state_dim, state_dim)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc1.bias.data.normal_(0.1)
        # self.fc2 = nn.Linear(64, 16)
        # self.fc2.weight.data.normal_(0, 0.1)
        # self.fc2.bias.data.normal_(0.1)
        self.fc3 = nn.Linear(state_dim, action_dim)
        self.fc3.weight.data.normal_(0, 0.1)
        self.fc3.bias.data.normal_(0.1)

    def forward(self, x):
        out = F.softmax(self.fc1(x), dim=1)
        out1 = x * out
        out2 = torch.sigmoid(self.fc3(out1))
        return out2

class Actor_Net(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Actor_Net, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc1.bias.data.normal_(0.1)
        self.fc2 = nn.Linear(64, 32)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc2.bias.data.normal_(0.1)
        self.fc3 = nn.Linear(32, action_dim)
        self.fc3.weight.data.normal_(0, 0.1)
        self.fc3.bias.data.normal_(0.1)

    def forward(self, x):
        out = torch.relu(self.fc1(x))
        out1 = torch.relu(self.fc2(out))
        out2 = torch.sigmoid(self.fc3(out1))
        return out2

class LSTM_Critic_Net(nn.Module):
    def __init__(self, state_dim, hidden_layer_size, action_dim):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(1, hidden_layer_size)
        self.fc1 = nn.Linear(hidden_layer_size * (state_dim + action_dim), action_dim)

    def forward(self, s, a):
        input_seq = torch.cat((s, a), 1)
        lstm_out, _ = self.lstm(torch.unsqueeze(input_seq, 2))
        predictions = self.fc1(lstm_out.view(lstm_out.shape[0], -1))
        # predictions = torch.relu(predictions)
        return predictions


class CNNnetwork(nn.Module):
    def __init__(self,):
        super().__init__()
        self.conv1d = nn.Conv1d(1,64,kernel_size=2)
        self.relu = nn.ReLU(inplace=True)
        self.Linear1= nn.Linear(64*11,50)
        self.Linear2= nn.Linear(50,1)
    def forward(self,x):
        x = self.conv1d(x)
        x = self.relu(x)
        x = x.view(-1)
        x = self.Linear1(x)
        x = self.relu(x)
        x = self.Linear2(x)
        return x

class LSTM_CNN_Critic_Net(nn.Module):
    def __init__(self, state_dim, hidden_layer_size, action_dim):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(1, hidden_layer_size)
        self.fc1 = nn.Linear(hidden_layer_size * (state_dim + action_dim), action_dim)
        self.cnn = nn.Sequential(nn.Conv1d(14, 20, 10, 5),
                                 nn.AvgPool1d(kernel_size=5, stride=3),
                                 nn.ReLU())

    def forward(self, s, a):
        input_seq = torch.cat((s, a), 1)
        lstm_out, _ = self.lstm(torch.unsqueeze(input_seq, 2))
        cnn_out = self.cnn(lstm_out)
        predictions = self.fc1(cnn_out.view(cnn_out.shape[0], -1))
        predictions = torch.relu(predictions)
        return predictions

class LSTM_Actor_Net(nn.Module):
    def __init__(self, state_dim, hidden_layer_size, action_dim):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(1, hidden_layer_size)
        self.fc1 = nn.Linear(hidden_layer_size * state_dim, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, input_seq):
        # print(input_seq.view(len(input_seq) ,1, -1).shape)
        lstm_out, _ = self.lstm(torch.unsqueeze(input_seq, 2))

        predictions = self.fc1(lstm_out.view(lstm_out.shape[0], -1))
        # predictions = torch.softmax(predictions, dim=1)
        # predictions = input_seq * predictions
        predictions = torch.sigmoid(self.fc2(predictions)) * 22.2
        return predictions

class LSTM_CNN_Actor_Net(nn.Module):
    def __init__(self, state_dim, hidden_layer_size, action_dim):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(1, hidden_layer_size)
        self.fc1 = nn.Linear(380, action_dim)
        self.cnn = nn.Sequential(nn.Conv1d(21, 20, 10, 5),
                                 nn.ReLU())

    def forward(self, input_seq):
        # x, b = input_seq.shape
        # print(torch.unsqueeze(input_seq, 2).shape)
        lstm_out, _ = self.lstm(torch.unsqueeze(input_seq, 2))
        # print(lstm_out.shape)
        cnn_out = self.cnn(lstm_out)
        # print(cnn_out.shape)
        predictions = self.fc1(cnn_out.view(cnn_out.shape[0], -1))
        predictions = torch.sigmoid(predictions) * 100
        return predictions

class Critic_Net(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic_Net, self).__init__()
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc1.bias.data.normal_(0.1)
        self.fc2 = nn.Linear(action_dim, 32)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc2.bias.data.normal_(0.1)
        self.fc3 = nn.Linear(32, 16)
        self.fc3.weight.data.normal_(0, 0.1)
        self.fc3.bias.data.normal_(0.1)
        self.fc4 = nn.Linear(16, 1)
        self.fc4.weight.data.normal_(0, 0.1)
        self.fc4.bias.data.normal_(0.1)

    def forward(self, s, a):

        x = self.fc1(s)
        y = self.fc2(a)
        out = F.relu(x + y)
        out1 = F.relu(self.fc3(out))
        out2 = self.fc4(out1)
        return out2

class DDPG_agent1(object):

    def __init__(self, state_counters, action_counters, batch_size, memory_size, LR_A=0.001, LR_C=0.001,
                 gamma=0.01, TAU=0.005):

        self.state_counters = state_counters
        self.action_counters = action_counters
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.TAU = TAU
        self.index_memory = 0
        self.memory = np.zeros((memory_size, self.state_counters * 2 + self.action_counters + 1))

        self.Actor_Net_eval, self.Actor_Net_target = Actor_Net(self.state_counters, self.action_counters), \
                                                     Actor_Net(self.state_counters, self.action_counters)

        self.Critic_Net_eval, self.Critic_Net_target = Critic_Net(self.state_counters, self.action_counters), \
                                                       Critic_Net(self.state_counters, self.action_counters)

        self.optimizer_A = torch.optim.Adam(self.Actor_Net_eval.parameters(), lr=LR_A)

        self.optimizer_C = torch.optim.Adam(self.Critic_Net_eval.parameters(), lr=LR_C)

        self.loss = nn.MSELoss()

    def choose_action(self, observation):

        observation = torch.FloatTensor(observation)
        observation = torch.unsqueeze(observation, 0)
        action = self.Actor_Net_eval(observation)[0].detach()

        return action.numpy()

    def store_memory(self, s, a, r, s_):

        memory = np.hstack((s, a, r, s_))
        index = self.index_memory % self.memory_size
        self.memory[index, :] = memory
        self.index_memory += 1

    def learn(self):

        sample_memory_index = np.random.choice(self.memory_size, self.batch_size)
        sample_memory = self.memory[sample_memory_index, :]
        sample_memory = torch.FloatTensor(sample_memory)
        sample_memory_s = sample_memory[:, : self.state_counters]
        sample_memory_s_ = sample_memory[:, - self.state_counters:]
        sample_memory_a = sample_memory[:, self.state_counters : self.state_counters + self.action_counters]
        sample_memory_r = sample_memory[:, - self.state_counters -1 : - self.state_counters]

        a = self.Actor_Net_target(sample_memory_s_)
        a_s = self.Actor_Net_eval(sample_memory_s)
        q_target = sample_memory_r + self.gamma * self.Critic_Net_target(sample_memory_s_, a)
        q_eval = self.Critic_Net_eval(sample_memory_s, sample_memory_a)
        loss_c = self.loss(q_target, q_eval)
        self.optimizer_C.zero_grad()
        loss_c.backward()
        self.optimizer_C.step()

        loss_a = - torch.mean(self.Critic_Net_eval(sample_memory_s, a_s))
        self.optimizer_A.zero_grad()
        loss_a.backward()
        self.optimizer_A.step()


        for parm, target_parm in zip(self.Actor_Net_eval.parameters(), self.Actor_Net_target.parameters()):
            target_parm.data.copy_(self.TAU * parm.data + (1 - self.TAU) * target_parm.data)
        for parm, target_parm in zip(self.Critic_Net_eval.parameters(), self.Critic_Net_target.parameters()):
            target_parm.data.copy_(self.TAU * parm.data + (1 - self.TAU) * target_parm.data)


class DDPG_agent_att(object):

    def __init__(self, state_counters, action_counters, batch_size, memory_size, LR_A=0.001, LR_C=0.001,
                 gamma=0.01, TAU=0.005):

        self.state_counters = state_counters
        self.action_counters = action_counters
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.TAU = TAU
        self.index_memory = 0
        self.memory = np.zeros((memory_size, self.state_counters * 2 + self.action_counters + 1))

        self.Actor_Net_eval, self.Actor_Net_target = Actor_Net_att(self.state_counters, self.action_counters), \
                                                     Actor_Net_att(self.state_counters, self.action_counters)

        self.Critic_Net_eval, self.Critic_Net_target = Critic_Net(self.state_counters, self.action_counters), \
                                                       Critic_Net(self.state_counters, self.action_counters)

        self.optimizer_A = torch.optim.Adam(self.Actor_Net_eval.parameters(), lr=LR_A)

        self.optimizer_C = torch.optim.Adam(self.Critic_Net_eval.parameters(), lr=LR_C)

        self.loss = nn.MSELoss()

    def choose_action(self, observation):

        observation = torch.FloatTensor(observation)
        observation = torch.unsqueeze(observation, 0)
        action = self.Actor_Net_eval(observation)[0].detach()

        return action.numpy()

    def store_memory(self, s, a, r, s_):

        memory = np.hstack((s, a, r, s_))
        index = self.index_memory % self.memory_size
        self.memory[index, :] = memory
        self.index_memory += 1

    def learn(self):

        sample_memory_index = np.random.choice(self.memory_size, self.batch_size)
        sample_memory = self.memory[sample_memory_index, :]
        sample_memory = torch.FloatTensor(sample_memory)
        sample_memory_s = sample_memory[:, : self.state_counters]
        sample_memory_s_ = sample_memory[:, - self.state_counters:]
        sample_memory_a = sample_memory[:, self.state_counters : self.state_counters + self.action_counters]
        sample_memory_r = sample_memory[:, - self.state_counters -1 : - self.state_counters]

        a = self.Actor_Net_target(sample_memory_s_)
        a_s = self.Actor_Net_eval(sample_memory_s)
        q_target = sample_memory_r + self.gamma * self.Critic_Net_target(sample_memory_s_, a)
        q_eval = self.Critic_Net_eval(sample_memory_s, sample_memory_a)
        loss_c = self.loss(q_target, q_eval)
        self.optimizer_C.zero_grad()
        loss_c.backward()
        self.optimizer_C.step()

        loss_a = - torch.mean(self.Critic_Net_eval(sample_memory_s, a_s))
        self.optimizer_A.zero_grad()
        loss_a.backward()
        self.optimizer_A.step()


        for parm, target_parm in zip(self.Actor_Net_eval.parameters(), self.Actor_Net_target.parameters()):
            target_parm.data.copy_(self.TAU * parm.data + (1 - self.TAU) * target_parm.data)
        for parm, target_parm in zip(self.Critic_Net_eval.parameters(), self.Critic_Net_target.parameters()):
            target_parm.data.copy_(self.TAU * parm.data + (1 - self.TAU) * target_parm.data)



class lstm_DDPG_agent2(object):

    def __init__(self, state_counters, action_counters, hidden_layer_size, batch_size, memory_size, LR_A=0.001, LR_C=0.001,
                 gamma=0.99, TAU=0.005):

        self.state_counters = state_counters
        self.action_counters = action_counters
        self.memory_size = memory_size
        self.hidden_layer_size = hidden_layer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.TAU = TAU
        self.index_memory = 0
        self.memory = np.zeros((memory_size, self.state_counters * 2 + self.action_counters + 1))

        self.Actor_Net_eval, self.Actor_Net_target = LSTM_Actor_Net(self.state_counters, self.hidden_layer_size, self.action_counters), \
                                                     LSTM_Actor_Net(self.state_counters, self.hidden_layer_size, self.action_counters)

        self.Critic_Net_eval, self.Critic_Net_target = Critic_Net(self.state_counters, self.action_counters), \
                                                       Critic_Net(self.state_counters, self.action_counters)

        self.optimizer_A = torch.optim.Adam(self.Actor_Net_eval.parameters(), lr=LR_A)

        self.optimizer_C = torch.optim.Adam(self.Critic_Net_eval.parameters(), lr=LR_C)
        self.optimizer_A.zero_grad()
        self.optimizer_C.zero_grad()
        self.loss = nn.MSELoss()

    def choose_action(self, observation):

        observation = torch.FloatTensor(observation)
        observation = torch.unsqueeze(observation, 0)
        action = self.Actor_Net_eval(observation)[0].detach()

        return action.numpy()

    def store_memory(self, s, a, r, s_):

        memory = np.hstack((s, a, r, s_))
        index = self.index_memory % self.memory_size
        self.memory[index, :] = memory
        self.index_memory += 1

    def learn1(self):

        sample_memory_index = np.random.choice(self.memory_size, self.batch_size)
        sample_memory = self.memory[sample_memory_index, :]
        sample_memory = torch.FloatTensor(sample_memory)
        sample_memory_s = sample_memory[:, : self.state_counters]
        sample_memory_s_ = sample_memory[:, - self.state_counters:]
        sample_memory_a = sample_memory[:, self.state_counters : self.state_counters + self.action_counters]
        sample_memory_r = sample_memory[:, - self.state_counters -1 : - self.state_counters]
        with torch.autograd.set_detect_anomaly(True):
            a = self.Actor_Net_target(sample_memory_s_)
            a_s = self.Actor_Net_eval(sample_memory_s)
            loss_a = - torch.mean(self.Critic_Net_eval(sample_memory_s, a_s))
            self.optimizer_A.zero_grad()
            self.optimizer_C.zero_grad()
            loss_a.backward()
            q_target = sample_memory_r + self.gamma * self.Critic_Net_target(sample_memory_s_, a)
            q_eval = self.Critic_Net_eval(sample_memory_s, sample_memory_a)
            # print(self.Critic_Net_eval.lstm.all_weights)
            # print("======")
            loss_c = self.loss(q_target, q_eval)
            loss_c.backward()
            self.optimizer_A.step()
            self.optimizer_C.step()
            for parm, target_parm in zip(self.Actor_Net_eval.parameters(), self.Actor_Net_target.parameters()):
                target_parm.data.copy_(self.TAU * parm.data + (1 - self.TAU) * target_parm.data)
            for parm, target_parm in zip(self.Critic_Net_eval.parameters(), self.Critic_Net_target.parameters()):
                target_parm.data.copy_(self.TAU * parm.data + (1 - self.TAU) * target_parm.data)

    def learn(self):

        sample_memory_index = np.random.choice(self.memory_size, self.batch_size)
        sample_memory = self.memory[sample_memory_index, :]
        sample_memory = torch.FloatTensor(sample_memory)
        sample_memory_s = sample_memory[:, : self.state_counters]
        sample_memory_s_ = sample_memory[:, - self.state_counters:]
        sample_memory_a = sample_memory[:, self.state_counters : self.state_counters + self.action_counters]
        sample_memory_r = sample_memory[:, - self.state_counters -1 : - self.state_counters]

        a = self.Actor_Net_target(sample_memory_s_)
        a_s = self.Actor_Net_eval(sample_memory_s)
        q_target = sample_memory_r + self.gamma * self.Critic_Net_target(sample_memory_s_, a)
        q_eval = self.Critic_Net_eval(sample_memory_s, sample_memory_a)
        loss_c = self.loss(q_target, q_eval)
        self.optimizer_C.zero_grad()
        loss_c.backward()
        self.optimizer_C.step()

        loss_a = - torch.mean(self.Critic_Net_eval(sample_memory_s, a_s))
        self.optimizer_A.zero_grad()
        loss_a.backward()
        self.optimizer_A.step()


        for parm, target_parm in zip(self.Actor_Net_eval.parameters(), self.Actor_Net_target.parameters()):
            target_parm.data.copy_(self.TAU * parm.data + (1 - self.TAU) * target_parm.data)
        for parm, target_parm in zip(self.Critic_Net_eval.parameters(), self.Critic_Net_target.parameters()):
            target_parm.data.copy_(self.TAU * parm.data + (1 - self.TAU) * target_parm.data)

class lstm_cnn_DDPG_agent2(object):

    def __init__(self, state_counters, action_counters, hidden_layer_size, batch_size, memory_size, LR_A=0.001, LR_C=0.001,
                 gamma=0.99, TAU=0.005):

        self.state_counters = state_counters
        self.action_counters = action_counters
        self.memory_size = memory_size
        self.hidden_layer_size = hidden_layer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.TAU = TAU
        self.index_memory = 0
        self.memory = np.zeros((memory_size, self.state_counters * 2 + self.action_counters + 1))

        self.Actor_Net_eval, self.Actor_Net_target = LSTM_CNN_Actor_Net(self.state_counters, self.hidden_layer_size, self.action_counters), \
                                                     LSTM_CNN_Actor_Net(self.state_counters, self.hidden_layer_size, self.action_counters)

        self.Critic_Net_eval, self.Critic_Net_target = Critic_Net(self.state_counters, self.action_counters), \
                                                       Critic_Net(self.state_counters, self.action_counters)

        self.optimizer_A = torch.optim.Adam(self.Actor_Net_eval.parameters(), lr=LR_A)

        self.optimizer_C = torch.optim.Adam(self.Critic_Net_eval.parameters(), lr=LR_C)
        self.optimizer_A.zero_grad()
        self.optimizer_C.zero_grad()
        self.loss = nn.MSELoss()

    def choose_action(self, observation):

        observation = torch.FloatTensor(observation)
        observation = torch.unsqueeze(observation, 0)
        action = self.Actor_Net_eval(observation)[0].detach()

        return action.numpy()

    def store_memory(self, s, a, r, s_):

        memory = np.hstack((s, a, r, s_))
        index = self.index_memory % self.memory_size
        self.memory[index, :] = memory
        self.index_memory += 1

    def learn1(self):

        sample_memory_index = np.random.choice(self.memory_size, self.batch_size)
        sample_memory = self.memory[sample_memory_index, :]
        sample_memory = torch.FloatTensor(sample_memory)
        sample_memory_s = sample_memory[:, : self.state_counters]
        sample_memory_s_ = sample_memory[:, - self.state_counters:]
        sample_memory_a = sample_memory[:, self.state_counters : self.state_counters + self.action_counters]
        sample_memory_r = sample_memory[:, - self.state_counters -1 : - self.state_counters]
        with torch.autograd.set_detect_anomaly(True):
            a = self.Actor_Net_target(sample_memory_s_)
            a_s = self.Actor_Net_eval(sample_memory_s)
            loss_a = - torch.mean(self.Critic_Net_eval(sample_memory_s, a_s))
            self.optimizer_A.zero_grad()
            self.optimizer_C.zero_grad()
            loss_a.backward()
            q_target = sample_memory_r + self.gamma * self.Critic_Net_target(sample_memory_s_, a)
            q_eval = self.Critic_Net_eval(sample_memory_s, sample_memory_a)
            # print(self.Critic_Net_eval.lstm.all_weights)
            # print("======")
            loss_c = self.loss(q_target, q_eval)
            loss_c.backward()
            self.optimizer_A.step()
            self.optimizer_C.step()
            for parm, target_parm in zip(self.Actor_Net_eval.parameters(), self.Actor_Net_target.parameters()):
                target_parm.data.copy_(self.TAU * parm.data + (1 - self.TAU) * target_parm.data)
            for parm, target_parm in zip(self.Critic_Net_eval.parameters(), self.Critic_Net_target.parameters()):
                target_parm.data.copy_(self.TAU * parm.data + (1 - self.TAU) * target_parm.data)

    def learn(self):

        sample_memory_index = np.random.choice(self.memory_size, self.batch_size)
        sample_memory = self.memory[sample_memory_index, :]
        sample_memory = torch.FloatTensor(sample_memory)
        sample_memory_s = sample_memory[:, : self.state_counters]
        sample_memory_s_ = sample_memory[:, - self.state_counters:]
        sample_memory_a = sample_memory[:, self.state_counters : self.state_counters + self.action_counters]
        sample_memory_r = sample_memory[:, - self.state_counters -1 : - self.state_counters]

        a = self.Actor_Net_target(sample_memory_s_)
        a_s = self.Actor_Net_eval(sample_memory_s)
        q_target = sample_memory_r + self.gamma * self.Critic_Net_target(sample_memory_s_, a)
        q_eval = self.Critic_Net_eval(sample_memory_s, sample_memory_a)
        loss_c = self.loss(q_target, q_eval)
        self.optimizer_C.zero_grad()
        loss_c.backward()
        self.optimizer_C.step()

        loss_a = - torch.mean(self.Critic_Net_eval(sample_memory_s, a_s))
        self.optimizer_A.zero_grad()
        loss_a.backward()
        self.optimizer_A.step()


        for parm, target_parm in zip(self.Actor_Net_eval.parameters(), self.Actor_Net_target.parameters()):
            target_parm.data.copy_(self.TAU * parm.data + (1 - self.TAU) * target_parm.data)
        for parm, target_parm in zip(self.Critic_Net_eval.parameters(), self.Critic_Net_target.parameters()):
            target_parm.data.copy_(self.TAU * parm.data + (1 - self.TAU) * target_parm.data)



