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

class Agent:

    def __init__(self):
        self.n_games = 0 # number of games played
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(38, 256, 1)
        self.trainer = QTrainer(self.model, learningRate=LR, gamma=self.gamma)

    def get_state(self, game):
        current_fruit = game.getCurrentFruit()
        positions = game.getPositions()
        if positions == []:
            positions = [[0] * 27] # there are 16 world matrix values
        for index in range(len(positions)):
            positions[index] = current_fruit + positions[index] # prepend current fruit to every position because we need uniform arrays
        # state = [current_fruit, positions]
        return np.array(positions)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # append a tuple

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

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
    
def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = suikasite.SuikaGame()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot results
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            
            if score > record:
                record = score
                agent.model.save()

            print("Game", agent.n_games, "Score", score, "Record", record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == "__main__":
    train()