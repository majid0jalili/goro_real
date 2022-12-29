import re
import threading
import os
import argparse
import multiprocessing
import torch
import time
import csv
import pandas as pd
from random import random
from random import randint
from datetime import datetime

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
num_features_per_core = 7

state_space = num_features_per_core*num_cpu
action_space = num_cpu
action_scale = pow(2, num_pf_per_core)

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

def run_mix(run_type, cmds, paths, mix_num):
    app = Applications(num_cpu)
    pebs = PEBS(num_cpu)
    pf = Prefetcher(num_cpu, num_pf_per_core)
   
    
    app.replay_runs(paths, cmds)
    
    actions = []
    instructions = []
    all_misses = []
    state, insts, llc_misses = pebs.state()
    
    bw = False
    
    while (True):
        tic = time.time()
        if(bw == False):
            bw = True
            print("-----Launched the BASE")
            app.run_bw("app_"+str(mix_num)+str("_bw_"+str(run_type)))
                
        if(run_type == "base"):
            pf.all_prefetchers_on()
            
        elif(run_type == "goro"): #RL
            action = agent.action(state, True)
            action = pf.all_prefetcher_set(action)
            actions.append(action)
                
        elif (run_type == "random") : # Random
            action = []
            acc_per_core = []
                
            for c in range(num_cpu):
                for pfr in range(num_pf_per_core):
                    acc_per_core.append(randint(0, 1))
                action.append(acc_per_core)
                acc_per_core = []
            action = pf.all_prefetcher_set(action)
            actions.append(action)
        
        state, insts, llc_misses = pebs.state()
        instructions.append(insts)
        all_misses.append(llc_misses)
        print(insts)
        wtime = 3 -((time.time()-tic))
        if(wtime > 0):
            time.sleep(wtime)
        if (app.num_running_apps() == 0):
            break
    
    duration = app.duration()
    app.kill_bw()
    
    return instructions, duration, actions, all_misses
    

        

def run_app(mix_num):
    run_mode = ["base", "goro", "random"]
    app = Applications(num_cpu)
    cmds = []
    paths = []
    
    for i in range(num_cpu):
        cmd_path, cmd_bg = app.get_spec_app(i*2)
        cmds.append(cmd_bg)
        paths.append(cmd_path)
    del app
    
    app_name = "app_"+str(mix_num)+".xlsx"
    df_cmds = pd.DataFrame(cmds)
    with pd.ExcelWriter(app_name) as writer:
        df_cmds.to_excel(writer, sheet_name="apps", index=False)
            
    
    for r_mode in run_mode:
        times = []
        times.append(datetime.now())
        instructions, duration, actions, all_misses = run_mix(r_mode, cmds, paths, mix_num)
        times.append(datetime.now())
        
        df_inst = pd.DataFrame(instructions)
        df_dur = pd.DataFrame(duration)
        df_act = pd.DataFrame(actions)
        df_all_misses = pd.DataFrame(all_misses)
        df_times = pd.DataFrame(times)
        
        with pd.ExcelWriter(app_name, mode='a') as writer:
            df_inst.to_excel(writer, sheet_name="inst_"+r_mode, index=False)
            df_dur.to_excel(writer, sheet_name="duration_"+r_mode, index=False)
            df_act.to_excel(writer, sheet_name="actions_"+r_mode, index=False)
            df_all_misses.to_excel(writer, sheet_name="misses_"+r_mode, index=False)
            df_times.to_excel(writer, sheet_name="times"+r_mode, index=False)

    
    
   
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
