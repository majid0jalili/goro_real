import re
import threading
import os
import argparse
import multiprocessing
import torch
import time
import csv

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


def run_app():
    app = Applications(num_cpu)
    while (True):
        for i in range(num_cpu):
            app.run_app()
        time.sleep(30)


def run_app(run_model):
    app = Applications(num_cpu)
    pebs = PEBS(num_cpu)
    pf = Prefetcher(num_cpu)

    cmds = []
    paths = []
    instructions = []
    for i in range(num_cpu):
        cmd_path, cmd_bg = app.run_app()
        cmds.append(cmd_bg)
        paths.append(cmd_path)

    pf.all_prefetchers_on()
    while (True):
        tic = time.time()
        inst = pebs.stats()
        instructions.append(inst)
        print(inst)
        wtime = 5-((time.time()-tic))
        time.sleep(wtime)
        if (app.num_running_apps() == 0):
            break
    with open(r'apps1.txt', 'w') as fp:
        for item in cmds:
            # write each item on a new line
            fp.write("%s\n" % item)

    with open("output1.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(instructions)
    app.replay_runs(paths, cmds)
    instructions = []
    while (True):
        tic = time.time()
        inst = pebs.stats()
        instructions.append(inst)
        print(inst)
        wtime = 5-((time.time()-tic))
        time.sleep(wtime)
        if (app.num_running_apps() == 0):
            break
    with open(r'apps2.txt', 'w') as fp:
        for item in cmds:
            # write each item on a new line
            fp.write("%s\n" % item)

    with open("output2.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(instructions)


def heartbeat():
    while (True):
        print("Loss:{} mem_size:{} beta:{} avg_reward:{}".format(
            loss, agent.memory.size(), agent.memory.beta, avg_reward))
        time.sleep(5)


def load_model():
    print("Function load_model")
    agent.load_model("./models/model", device)


def main():
    load_model()

    run_app(False)

    return

    t_run_app = threading.Thread(target=run_app, args=())
    t_heartbeat = threading.Thread(target=heartbeat, args=())

    t_set_collector.start()
    t_heartbeat.start()

    t_set_collector.join()
    t_heartbeat.join()

    # all threads completely executed
    print("Done!")


if __name__ == '__main__':
    main()
