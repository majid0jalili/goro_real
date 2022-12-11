import re
import threading
import os
import argparse
import multiprocessing
import torch
import time

from concurrent.futures import ProcessPoolExecutor
from torch.multiprocessing import Process, set_start_method, Queue
import asyncio


# BDQ
from agent import BQN
from benches import Applications
from pbes import PEBS
from prefetcher import Prefetcher

parser = argparse.ArgumentParser('parameters')
parser.add_argument('--lr_rate', type=float, default=1e-3,
                    help='learning rate (default : 0.0001)')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size(default : 32)')
parser.add_argument('--gamma', type=float, default=0.98,
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

alpha = 0.2
beta = 0.6


device = 'cuda' if torch.cuda.is_available() else 'cpu'




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


def train(agent):
    print("Function set_collector")
    pebs = PEBS(num_cpu)
    pf = Prefetcher(num_cpu, num_pf_per_core)
    app = Applications(num_cpu)
    total_reward = 0
    state, insts = pebs.state()
    next_state = []
    next_inst = []
    avg_reward = 0
    itr = 0
    loss = 0
    elapsed = {}
   
    while (True):
        itr += 1
        for i in range(num_cpu):
            app.run_app()
        
        
        tic = time.time()
        action = agent.action(state)
        toc = time.time()
        elapsed["action"]=(toc-tic)
        
        tic = time.time()
        action = pf.all_prefetcher_set(action)
        toc = time.time()
        elapsed["pf_Set"]=(toc-tic)
        
        tic = time.time()
        next_state, next_inst = pebs.state()
        toc = time.time()
        elapsed["read_state"]=(toc-tic)
        
        
        
        reward = 0
        for inst in range(len(insts)):
            if (insts[inst] != 0):
                if ((next_inst[inst] / insts[inst]) - 1 < 4 and (next_inst[inst] / insts[inst]) - 1 > -4):
                    reward += (next_inst[inst] / insts[inst]) - 1

        r_arr = [reward]
        total_reward += reward
        
        
        if (itr == 100):
            print(elapsed)
            avg_reward = total_reward / 100
            
            with open(r'./avg_reward.txt', 'a') as fp:
                fp.write('avg_reward ' + str(avg_reward) +
                         ' loss '+str(loss)+
                          '\n')
            itr = 0
            total_reward = 0
            fp.close()
            agent.memory.write_to_csv("mem.xlsx")
            agent.save_model("model")
            agent.memory.beta = beta

   
        agent.memory.write_buffer(state, next_state, action, r_arr)

        if (agent.memory.size() > 8*batch_size):
            tic = time.time()
            loss = agent.train_model(batch_size, gamma)
            agent.memory.beta = beta + (itr/100)*(1-beta)
            toc = time.time()
            elapsed["train"]=(toc-tic)

       
   
        state = next_state
        insts = next_inst

   


def load_model(agent):
    print("Function load_model")
    agent.load_model("./models/model", device)


def main():
    agent = BQN(state_space, action_space, action_scale,
            learning_rate, device, num_cpu, num_pf_per_core, alpha, beta)
      
    train(agent)
if __name__ == '__main__':
    main()
    