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

num_cpu = 64
num_pf_per_core = 1
num_features_per_core = 7

# state_space = num_features_per_core*num_cpu
state_space = num_cpu*16
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

def make_action(run_type, state, num_app):
    action_start = time.time()
    actions = []
    
    if(run_type == "base" or num_app < (0.75*num_cpu)):
        action = []
        acc_per_core = []
        for c in range(num_cpu):
            for pfr in range(num_pf_per_core):
                acc_per_core.append(0)
            action.append(acc_per_core)
            acc_per_core = []
        actions = action
        
    elif(run_type == "goro"): #RL
        action = agent.action(state, True)
        actions = action
    elif (run_type == "random") : # Random
        action = []
        acc_per_core = []
        for c in range(num_cpu):
            for pfr in range(num_pf_per_core):
                acc_per_core.append(randint(0, 1))
            action.append(acc_per_core)
            acc_per_core = []
        actions = action
    action_elapsed = round(time.time() - action_start, 2)
    
    return actions, action_elapsed
       
    
def take_action(actions, pf):
    start = time.time()
    action = pf.all_prefetcher_set(actions)
    end = time.time()
    return round(end-start, 2)
    
def measure_pebs(pebs):
    time.sleep(0.1)
    start = time.time()
    state, insts, llc_misses = pebs.state1()
    end = time.time()
    return state, insts, round(end-start, 2)


def run_app(mix_num):
    # run_mode = ["base", "goro", "random"]
    run_mode = ["goro", "base", "random"]
    
    
    app = Applications(num_cpu)
    app.run_perf_stat()
    print("Initlize wait for test.csv")
    time.sleep(5) 
    pf = Prefetcher(num_cpu, num_pf_per_core)
    
    cmds = []
    paths = []
    
    
    pebs = PEBS(num_cpu)
    state, insts, llc_misses = pebs.state1()
    
    
    
    for i in range(num_cpu):
        cmd_path, cmd_bg = app.get_spec_app(i*2)
        cmds.append(cmd_bg)
        paths.append(cmd_path)
    
    
    app_name = "app_"+str(mix_num)+".xlsx"
    df_cmds = pd.DataFrame(cmds)
    with pd.ExcelWriter(app_name) as writer:
        df_cmds.to_excel(writer, sheet_name="apps", index=False)
            
 
   
    
    for r_mode in run_mode:
        app.replay_runs(paths, cmds)
        instructions = []
        actions = []
        itr = 0
        app.run_bw("app_"+str(mix_num)+str("_bw_"+str(r_mode)))
        num_app = app.num_running_apps()
        while(num_app != 0):
            action, make_action_length = make_action(r_mode, state, num_app)
            take_action_length = take_action(action, pf)
            state, insts, pebs_length = measure_pebs(pebs)
            
            instructions.append(insts)
            actions.append(action)

            if(itr == 100):
                itr = 0
                print("{} take_action:{} make_action:{} pebs_length:{}".format(r_mode, take_action_length, make_action_length, pebs_length)) 
                print(app.core_PID)
            itr += 1
            num_app = app.num_running_apps()
            
        duration = app.duration()
        app.kill_bw()
        app.app_rest()
        app.run_perf_stat()
        time.sleep(5)
        
        df_inst = pd.DataFrame(instructions)
        df_dur = pd.DataFrame(duration)
        df_act = pd.DataFrame(actions)


        
        with pd.ExcelWriter(app_name, engine="openpyxl", mode='a') as writer:
            df_inst.to_excel(writer, sheet_name="inst_"+r_mode, index=False)
            df_dur.to_excel(writer, sheet_name="duration_"+r_mode, index=False)
            df_act.to_excel(writer, sheet_name="actions_"+r_mode, index=False)


    # app.kill_bw()
    app.kill_perf_stat()
    
   
def load_model():
    print("Function load_model")
    agent.load_model("./models/model", device)


def main():
    load_model()

    for i in range(0, 12, 1):
        run_app(i)

    return

if __name__ == '__main__':
    main()
