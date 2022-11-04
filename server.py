import re
import threading
import os
import argparse
import multiprocessing
import torch
import time

# BDQ
from utils import ReplayBuffer
from agent import BQN
from benches import Applications
from pbes import PEBS
from prefetcher import Prefetcher

parser = argparse.ArgumentParser('parameters')
parser.add_argument('--lr_rate', type=float, default=1e-3,
                    help='learning rate (default : 0.0001)')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size(default : 32)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='gamma (default : 0.99)')
parser.add_argument("--name", type=str, default='unknown')
parser.add_argument("--mlmode", type=str, default='train')

args = parser.parse_args()

learning_rate = args.lr_rate
batch_size = args.batch_size
gamma = args.gamma
run_name = args.name
mlmode = args.mlmode

num_cpu = 16
num_pf_per_core = 4
num_features_per_core = 6

state_space = num_features_per_core*num_cpu
action_space = num_pf_per_core*num_cpu
action_scale = 2
total_reward = 0

loss = 0

device = 'cuda' if torch.cuda.is_available() else 'cpu'
memory = ReplayBuffer(1000, action_space, device)
agent = BQN(state_space, action_space, action_scale,
            learning_rate, device, num_cpu, num_pf_per_core)


def summary():
    print("cpu_count", multiprocessing.cpu_count())
    print("state_space", state_space)
    print("action_space", action_space)
    print("action_scale", action_scale)
    print("total_reward", total_reward)
    print("learning_rate", learning_rate)
    print("batch_size", batch_size)
    print("gamma", gamma)
    print("run_name", run_name)
    print("mlmode", mlmode)
    print("----------------")


def run_app():
    app = Applications(num_cpu)
    while (True):
        for i in range(num_cpu):
            app.run_app()
        time.sleep(30)


def set_collector():
    print("Function set_collector")
    pebs = PEBS(num_cpu)
    pf = Prefetcher(num_cpu)
    total_reward = 0
    state, insts = pebs.state()
    next_state = []
    next_inst = []
    reward = 0
    avg_reward = 0
    itr = 0

    while (True):
        itr += 1
        action = agent.action(state)
        action = pf.all_prefetcher_set(action)
        next_state, next_inst = pebs.state()
        reward = 0
        for inst in range(len(insts)):
            if (insts[inst] != 0):
                if ((next_inst[inst] / insts[inst]) - 1 < 2 and (next_inst[inst] / insts[inst]) - 1 > -2):
                    reward += (next_inst[inst] / insts[inst]) - 1

        if (reward < -2):
            reward = -4
        elif (reward < -1):
            reward = -3
        elif (reward < -.5):
            reward = -2
        elif (reward < 0):
            reward = -1
        elif (reward < .5):
            reward = 1
        elif (reward < 1):
            reward = 2
        elif (reward < 2):
            reward = 3
        else:
            reward = 4

        r_arr = [reward]
        total_reward += reward
        avg_reward += reward

        if (itr == 100):
            avg_reward = avg_reward / 100
            with open(r'./avg_reward.txt', 'a') as fp:
                fp.write('avg_reward ' + str(avg_reward) +
                         ' loss '+str(loss) +
                         '\n')
            itr = 0
            avg_reward = 0
            fp.close()

        memory.write_buffer(state, next_state, action, r_arr)

        state = next_state
        insts = next_inst


def train():
    loss_itr = 0
    train_itr = 0
    global loss
    print("Function train")
    while True:
        if (memory.size() > batch_size):
            loss = agent.train_model(memory, batch_size, gamma)
            loss_itr += 1
            train_itr += 1
            if (loss_itr == 500):
                print("train_itr Loss:", train_itr, loss.item())
                agent.save_model("model")
                loss_itr = 0


def load_model():
    print("Function load_model")
    agent.load_model("./models/model", device)


def main():
    # summary()

    # creating thread
    t_run_app = threading.Thread(target=run_app, args=())
    t_set_collector = threading.Thread(target=set_collector, args=())
    t_train = threading.Thread(target=train, args=())

    if (mlmode == "train"):
        t_train.start()
        t_run_app.start()
    else:
        load_model()
    t_set_collector.start()

    t_set_collector.join()
    if (mlmode == "train"):
        t_train.join()
        t_run_app.join()

    # all threads completely executed
    print("Done!")


if __name__ == '__main__':
    main()
