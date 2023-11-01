import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class DeepQNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DeepQNetwork, self).__init__()

        self.conv1 = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Linear(hidden_size, output_size))

        self._create_weights()
        self.double()
        # self.linear1 = nn.Linear(input_size, hidden_size)
        # self.linear2 = nn.Linear(hidden_size, output_size)

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
    
    def save(self, record, n_games, file_name="model.pth"): #TODO write high score to file as well
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        recordF = os.path.join(model_folder_path, "history")
        with open(recordF, "w") as f:
            print("writing record: ", record)
            print("record type: ", type(record))
            f.write("[], []".format(str(record), str(n_games)))
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, learningRate, gamma) -> None:
        self.learningRate = learningRate
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.learningRate)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float) # seems to be expecting a 1d array
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.int)
        # (n, x)

        if len(state.shape) == 2:
            # (1, x)
            # print("Squeezed state: ", state)
            # state = torch.unsqueeze(state, 0)
            # print("Unsqueezed state: ", state)
            # next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: get predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action).item()] = Q_new
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad() # empty the gradient
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()