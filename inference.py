import re
import threading
import os
import argparse
import multiprocessing
import torch
import time
import csv
import pandas as pd

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

def run_mix(run_type, paths, cmds):
    app = Applications(num_cpu)
    pebs = PEBS(num_cpu)
    pf = Prefetcher(num_cpu)
    cmds = []
    paths = []
    instructions = []
    
    app.replay_runs(paths, cmds)
    actions = []
    state, insts = pebs.state()
    
    while (True):
        tic = time.time()
        if(run_type == 0):
            pf.all_prefetchers_on()
        elif(run_type == 1): #RL
            action = agent.action(state)
            action = pf.all_prefetcher_set(action)
        elif (run_type == 2) : # Random
            action = []
            acc_per_core = []
            for c in range(num_cpu):
                for pf in range(num_pf_per_core):
                    acc_per_core.append(randint(0, 1))
                action.append(acc_per_core)
                acc_per_core = []
            action = pf.all_prefetcher_set(action)
        
        state, insts = pebs.state()
        instructions.append(insts)
        print(insts)
        wtime = 5-((time.time()-tic))
        time.sleep(wtime)
        if (app.num_running_apps() == 0):
            break
    
    duration = app.duration()
    
    return instructions, duration
    

        

def run_app(mix_num):
    app = Applications(num_cpu)
    pebs = PEBS(num_cpu)
    pf = Prefetcher(num_cpu)
    
    
    cmds = []
    paths = []
    instructions_base = []
    for i in range(num_cpu):
        cmd_path, cmd_bg = app.run_app()
        cmds.append(cmd_bg)
        paths.append(cmd_path)

    pf.all_prefetchers_on()
    while (True):
        tic = time.time()
        state, insts = pebs.state()
        instructions_base.append(insts)
        print(insts)
        wtime = 5-((time.time()-tic))
        time.sleep(wtime)
        if (app.num_running_apps() == 0):
            break
    
    duration_base = app.duration()
    app.replay_runs(paths, cmds)
    
    instructions = []
    actions = []
    state, insts = pebs.state()
    while (True):
        tic = time.time()
        action = agent.action(state)
        action = pf.all_prefetcher_set(action)
        state, insts = pebs.state()
        instructions.append(insts)
        actions.append(action)
        print(insts)
        wtime = 5-((time.time()-tic))
        time.sleep(wtime)
        if (app.num_running_apps() == 0):
            break

    duration = app.duration()

    app_name = "app_"+str(mix_num)+".xlsx"
    df_app = pd.DataFrame(cmds)
    df_duration_base = pd.DataFrame(duration_base)
    df_duration = pd.DataFrame(duration)
    df_instructions_base = pd.DataFrame(instructions_base)
    df_instructions = pd.DataFrame(instructions)
    df_actions = pd.DataFrame(actions)
    
        
    with pd.ExcelWriter(app_name) as writer:
        df_app.to_excel(writer, sheet_name="apps", index=False)
        df_duration_base.to_excel(writer, sheet_name="duration_base", index=False)
        df_duration.to_excel(writer, sheet_name="duration", index=False)
        df_instructions_base.to_excel(writer, sheet_name="instructions_base", index=False)
        df_instructions.to_excel(writer, sheet_name="instructions", index=False)
        df_actions.to_excel(writer, sheet_name="actions", index=False)
    
    
   
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

    for i in range(64):
        run_app(i)

    return

if __name__ == '__main__':
    main()
