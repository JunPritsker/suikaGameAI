"""
previous @author: Viet Nguyen <nhviet1009@gmail.com>
adapted by @author: Jun Pritsker <junpritsker@gmail.com>
"""
import argparse
import os
import shutil
from random import random, randint, sample

import numpy as np
import torch
import torch.nn as nn

from model import DeepQNetwork
from collections import deque

import suikasite
from helper import plot

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play suikagame""")
    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--batch_size", type=int, default=512, help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--num_decay_epochs", type=float, default=6)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=2)
    parser.add_argument("--replay_memory_size", type=int, default=1000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    model = DeepQNetwork(27,256,1) #16 inputs from network
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()

    game = suikasite.SuikaGame() # this already "restarts the game, no need to restart below
    # state will be current piece in hand, game state(fruits and positions), and possible actions: list(range(-215,215,1))
    state = game.getState()
    if torch.cuda.is_available():
        model.cuda()
        state = state.cuda()

    replay_memory = deque(maxlen=opt.replay_memory_size)
    epoch = 0
    plotScores = []
    plotMeanScores = []
    totalScore = 0
    while epoch < opt.num_epochs:
        moves = 0
        next_steps = game.getNextStates()
        # print("next steps: ", next_steps)
        # Exploration or exploitation
        epsilon = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) * (
                opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)
        u = random()
        random_action = u <= epsilon
        # for item, value in zip(next_steps):
            # print("ZIP: ", item, value)
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        if torch.cuda.is_available():
            next_states = next_states.cuda()
        model.eval()
        with torch.no_grad():
            predictions = model(next_states)[:, 0]
        model.train()
        if random_action:
            index = randint(0, len(next_steps) - 1)
        else:
            index = torch.argmax(predictions).item()

        next_state = next_states[index, :]
        action = next_actions[index]

        reward, done, score = game.playStep(action)
        moves += 1
        print("Action: {} Reward: {}".format(action, reward))

        if torch.cuda.is_available():
            next_state = next_state.cuda()
        replay_memory.append([state, reward, score, next_state, done]) # determine sequence of when we calculate rewards. may need to wait for balls to start moving for accurate reward but this costs time
        if done:
            final_score = game.score
            totalScore += final_score
            plotScores.append(final_score)
            # final_tetrominoes = env.tetrominoes
            # final_cleared_lines = env.cleared_lines
            # state = env.reset()
            state = game.restartGame()
            if torch.cuda.is_available():
                state = state.cuda()
        else:
            # print("NOT DONE")
            state = next_state
            continue
        if len(replay_memory) < opt.replay_memory_size / 10:
            print("Replay memory size: {}".format(len(replay_memory)))
            continue
        epoch += 1
        print("EPOCH: {}".format(epoch))
        meanScore = totalScore / epoch
        plotMeanScores.append(meanScore)
        plot(plotScores, plotMeanScores)

        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_batch, reward_batch, score_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.cat(tuple(state for state in state_batch))
        print("State batch: {}, len: {}".format(state_batch, len(state_batch)))
        score_batch = torch.from_numpy(np.array(score_batch, dtype=np.float32)[:, None])
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        print("Reward batch: {}, len: {}".format(reward_batch, len(reward_batch)))
        next_state_batch = torch.cat(tuple(state for state in next_state_batch))

        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()

        q_values = model(state_batch)
        model.eval()
        with torch.no_grad():
            next_prediction_batch = model(next_state_batch)
        model.train()

        y_batch = torch.cat(
            tuple(reward if done else reward + opt.gamma * prediction for reward, done, prediction in
                  zip(reward_batch, done_batch, next_prediction_batch)))[:, None]

        optimizer.zero_grad()
        print("qval: {}, y_batch: {}".format(q_values, y_batch))
        print("lens qval: {}, y_batch: {}".format(len(q_values), len(y_batch)))
        loss = criterion(q_values, y_batch)
        loss.backward()
        optimizer.step()

        print("Epoch: {}/{}, Action: {}, Score: {}".format(
            epoch,
            opt.num_epochs,
            action,
            final_score))
        # writer.add_scalar('Train/Score', final_score, epoch - 1)
        # writer.add_scalar('Train/Tetrominoes', final_tetrominoes, epoch - 1)
        # writer.add_scalar('Train/Cleared lines', final_cleared_lines, epoch - 1)

        if epoch > 0 and epoch % opt.save_interval == 0:
            torch.save(model, "{}/suika_{}".format(opt.saved_path, epoch))

    torch.save(model, "{}/suika".format(opt.saved_path))


if __name__ == "__main__":
    opt = get_args()
    train(opt)