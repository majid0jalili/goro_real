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

parser = argparse.ArgumentParser('parameters')
parser.add_argument('--lr_rate', type=float, default=1e-4,
                    help='learning rate (default : 0.0001)')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size(default : 64)')
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


state_space = 28
action_space = 20
action_scale = 2
total_reward = 0

device = 'cuda' if torch.cuda.is_available() else 'cpu'
memory = ReplayBuffer(1000, action_space, device)
# agent = BQN(state_space, action_space, action_scale, learning_rate, device)


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
    app = Applications(4)
    while(True):
        app.run_app()
        app.run_app()
        app.run_app()
        app.run_app()
        
        time.sleep(5)


def set_collector():
    print("Function set_collector")
    
    while(True):
        pebs = PEBS(4)
        stats = pebs.run_perf_stat()
        pebs = pebs.print(stats)
        time.sleep(5)


def train():
    print("Function train")


def load_model():
    print("Function load_model")


def main():
    # summary()

    # creating thread
    t_run_app = threading.Thread(target=run_app, args=())
    t_set_collector = threading.Thread(target=set_collector, args=())
    t_train = threading.Thread(target=train, args=())

    t_run_app.start()
    t_set_collector.start()
    if (mlmode == "train"):
        t_train.start()
    else:
        load_model()

    t_run_app.join()
    t_set_collector.join()
    if (mlmode == "train"):
        t_train.join()

    # all threads completely executed
    print("Done!")


if __name__ == '__main__':
    main()
