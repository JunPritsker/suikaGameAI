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
        self.float()
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