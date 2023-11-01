import os
import csv
import random
import suikasite
import numpy as np
import torch
from collections import deque
from model import DeepQNetwork, QTrainer
from helper import plot
import train

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

os.environ["DISPLAY"] = ":0" #set display for WSL

class Agent:

    def __init__(self):
        self.n_games = 0 # number of games played
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = DeepQNetwork(16, 256, 1)
        self.loadedModel, self.record, self.n_games = self.loadModel()
        self.model = self.loadedModel if not self.loadedModel == None else self.model
        self.record = self.record if not self.record == None else 0 # record high score
        self.n_games = self.n_games if not self.n_games == None else 0
        self.trainer = QTrainer(self.model, learningRate=LR, gamma=self.gamma)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # append a tuple

    def get_action(self, state):
        # random moves: tradeoff between exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = None
        ###
        # if epsilon isnt' high enough, we go with a random move
        # as epsilon decreases (as we have more games played), we'll start making calculated moves instead to
        # improve the exploitation of the model
        if random.randint(0,200) < self.epsilon:
            final_move = random.randint(-215,215)
            print("Random move: ", final_move)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move = move
            print("Intentional move: ", move)

        return final_move
    
    def loadModel(self, file_name="model.pth"):
        model_folder_path = "./model"
        if os.path.exists(model_folder_path):
            file_name = os.path.join(model_folder_path, file_name)
            model = self.model
            model.load_state_dict(torch.load(file_name))
            historyF = os.path.join(model_folder_path, "history")
            if os.path.exists(historyF):
                with open(historyF, "r") as f:
                    csvFile = csv.reader(f)
                    for line in csvFile:
                        record, n_games = line
                print("loaded record: ", record, " | ngames: ", n_games)
            else:
                record, n_games = 0, 0
            print("model and record loaded")
            return model, int(record), int(n_games)
        return None, None, None

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    agent = Agent()
    game = suikasite.SuikaGame()
    record = agent.record
    while True:
        if not game.checkGameOver():
            # get old state
            state_old = game.get_state()

            # get move
            final_move = agent.get_action(state_old)

            # perform move and get new state
            reward, done, score = game.play_step(final_move)
            # print("Score: ", score, " | Reward: ", reward)
            try:
                state_new = game.get_state()
            except: #Game ended at a weird time, use old state
                state_new = state_old
                reward = -10
                done = True

            # train short memory
            agent.train_short_memory(state_old, final_move, reward, state_new, done)

            # remember
            agent.remember(state_old, final_move, reward, state_new, done)
        else:
            done = True

        if done:
            # train long memory, plot results
            game.restartGame()
            agent.n_games += 1
            agent.train_long_memory()
            
            if score > record:
                record = score
                agent.model.save(record, agent.n_games)

            print("Game", agent.n_games, "Score", score, "Record", record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

# if __name__ == "__main__":
    # train()