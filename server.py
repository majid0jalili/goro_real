import re
import threading
import os
import argparse
import multiprocessing
import torch
import time

# BDQ
from agent import BQN
from benches import Applications
from pbes import PEBS
from prefetcher import Prefetcher

parser = argparse.ArgumentParser('parameters')
parser.add_argument('--lr_rate', type=float, default=1e-3,
                    help='learning rate (default : 0.0001)')
parser.add_argument('--batch_size', type=int, default=16,
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

alpha = 0.2
beta = 0.6
loss = 0
avg_reward = 0


device = 'cuda' if torch.cuda.is_available() else 'cpu'

agent = BQN(state_space, action_space, action_scale,
            learning_rate, device, num_cpu, num_pf_per_core, alpha, beta)


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
    global avg_reward
    itr = 0

    while (True):
        itr += 1
        tic = time.time()
        state = torch.rand((1, state_space)).float().to(device)
        action = agent.action(state)
        toc = time.time()
        print("Length", toc-tic)
        action = pf.all_prefetcher_set(action)
        next_state, next_inst = pebs.state()
        reward = 0
        for inst in range(len(insts)):
            if (insts[inst] != 0):
                if ((next_inst[inst] / insts[inst]) - 1 < 4 and (next_inst[inst] / insts[inst]) - 1 > -4):
                    reward += (next_inst[inst] / insts[inst]) - 1

        r_arr = [reward]
        total_reward += reward

        if (itr == 100):
            avg_reward = total_reward / 100
            
            with open(r'./avg_reward.txt', 'a') as fp:
                fp.write('avg_reward ' + str(avg_reward) +
                         ' loss '+str(loss) +
                         '\n')
            itr = 0
            total_reward = 0
            fp.close()

            agent.memory.write_to_csv("mem.xlsx")

        agent.memory.write_buffer(state, next_state, action, r_arr)

        state = next_state
        insts = next_inst


def heartbeat():
    while (True):
        print("Loss:{} mem_size:{} beta:{} avg_reward:{}".format(
            loss, agent.memory.size(), agent.memory.beta, avg_reward))
        time.sleep(5)


def train():
    loss_itr = 0
    train_itr = 0
    global loss
    print("Function train")
    while True:
        if (agent.memory.size() > 10*batch_size):
            loss = agent.train_model(batch_size, gamma)
            loss_itr += 1
            train_itr += 1
            agent.memory.beta = beta + (loss_itr/100)*(1-beta)
            if (loss_itr == 100):
                # print("train_itr Loss:", train_itr, loss.item())
                agent.save_model("model")
                loss_itr = 0
                agent.memory.beta = beta


def load_model():
    print("Function load_model")
    agent.load_model("./models/model", device)


def main():
    # summary()

    # creating thread
    t_run_app = threading.Thread(target=run_app, args=())
    t_set_collector = threading.Thread(target=set_collector, args=())
    t_train = threading.Thread(target=train, args=())
    t_heartbeat = threading.Thread(target=heartbeat, args=())

    if (mlmode == "train"):
        t_train.start()
        t_run_app.start()
        t_heartbeat.start()
    else:
        load_model()
    t_set_collector.start()

    t_set_collector.join()
    if (mlmode == "train"):
        t_train.join()
        t_run_app.join()
        t_heartbeat.join()

    # all threads completely executed
    print("Done!")


if __name__ == '__main__':
    main()
