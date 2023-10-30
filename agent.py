import os
import random
import suikasite
import numpy as np
import torch
from collections import deque
from model import Linear_QNet, QTrainer
from helper import plot

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
        self.model = Linear_QNet(38, 256, 1)
        self.loadedModel, self.record = self.loadModel()
        self.model = self.loadedModel if not self.loadedModel == None else self.model
        self.record = self.record if not self.record == None else 0
        self.trainer = QTrainer(self.model, learningRate=LR, gamma=self.gamma)

    def get_state(self, game):
        # current_fruit = game.pauseAngGetData(game.getCurrentFruit())
        # positions = game.pauseAndGetData(game.getPositions())
        current_fruit, positions = game.pauseAndGetData((game.getCurrentFruit(), game.getPositions()))
        if positions == []:
            positions = [[0] * 27] # there are 16 world matrix values
        for index in range(len(positions)):
            positions[index] = current_fruit + positions[index] # prepend current fruit to every position because we need uniform arrays
        # state = [current_fruit, positions]
        return np.array(positions, dtype=float)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # append a tuple

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        # print("States type: ", type(states))
        # print("States size: ", len(states))
        # print("States size 1: ", len(states[0]))
        # print("States size 1[0]: ", len(states[0][0]))
        # print("States size 15: ", len(states[15]))
        # print("States 1", states[0])
        # print("States 1 shape", states[0].shape)
        # print("States 15", states[15])
        # print("States 15 shape", states[15].shape)
        # print("rewards size: ", len(rewards))
        # print("rewards[0] size: ", len(rewards[0]))
        # print("rewards[0]: ", rewards[0])
        for state, action, reward, next_state, done in mini_sample: #this method is much slower TODO: fix batching for fast pytorch processing
            self.trainer.train_step(state, action, reward, next_state, done)
        # self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

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
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move = move

        return final_move
    
    def loadModel(self, file_name="model.pth"):
        model_folder_path = "./model"
        if os.path.exists(model_folder_path):
            file_name = os.path.join(model_folder_path, file_name)
            model = self.model
            model.load_state_dict(torch.load(file_name))
            recordF = os.path.join(model_folder_path, "record")
            with open(recordF, "r"):
                record = int(recordF.readlines())
                print("loaded record: ", record)
            print("model and record loaded")
            return model, record
        return None, None

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
            state_old = agent.get_state(game)

            # get move
            final_move = agent.get_action(state_old)

            # perform move and get new state
            reward, done, score = game.play_step(final_move)
            print("Score: ", score, " | Reward: ", reward)
            try:
                state_new = agent.get_state(game)
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
                agent.model.save(record)

            print("Game", agent.n_games, "Score", score, "Record", record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == "__main__":
    train()