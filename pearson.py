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
import numpy as np

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
features = 50
num_pf_per_core = 1

alpha = 0.2
beta = 0.6
loss = 0
avg_reward = 0
all_events = []
evensts = [] 

all_values = pd.DataFrame()                   
all_values_abs = pd.DataFrame()
last_values = np.zeros(shape=(num_cpu, features+2))


first_done = 0


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
    global all_values, first_done, last_values, all_values_abs
    
    time.sleep(0.1)
    start = time.time()
    state, insts, state_np = pebs.state1(evensts)
    state_np = state_np.astype(float)

    # calculate the element-wise division of the last two columns
    div = state_np[:, -2] / state_np[:, -1]
    # replace the last two columns with the division result
    state_np[:, -2:] = div.reshape((-1, 1))

    if(first_done > 0):
        a_sub = state_np[:, :-2]
        # subtract the last two from the last two of the second numpy
        b_sub = state_np[:, -2:] - last_values[:, -2:] 
        # concatenate the two arrays horizontally
        result = np.hstack((a_sub, b_sub))
        all_values = pd.concat([all_values, pd.DataFrame(result, columns=evensts)], ignore_index=True)
        # all_values = pd.concat([all_values, pd.DataFrame(state_np, columns=evensts)], ignore_index=True)
       
    
    all_values_abs = pd.concat([all_values_abs, pd.DataFrame(state_np, columns=evensts)], ignore_index=True)
    last_values = state_np.astype(float)    
  
    first_done += 1
    
    # all_values_abs.to_csv("all_values_abs.csv")
    # all_values.to_csv("all_values.csv")
    end = time.time()
    return state, insts, round(end-start, 2)


def run_app(mix_num, name):
    run_mode = ["random"]
    
    event_list = ""
    for e in evensts:
        event_list +=e+","
    event_list = event_list[:-1]
    
    app = Applications(num_cpu)
    app.run_perf_stat(event_list)
    print("Initlize wait for test.csv")
    time.sleep(5) 
    pf = Prefetcher(num_cpu, num_pf_per_core)
    
    cmds = []
    paths = []
    
    
    pebs = PEBS(num_cpu)
    state, insts, state_dict = pebs.state1(evensts)
    
    
    
    for i in range(num_cpu):
        cmd_path, cmd_bg = app.get_spec_app(i)
        cmds.append(cmd_bg)
        paths.append(cmd_path)
       
    df_cmds = pd.DataFrame(cmds)
    
    for r_mode in run_mode:
        app.replay_runs(paths, cmds)
        instructions = []
        actions = []
        itr = 0
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
        app.run_perf_stat(event_list)
        time.sleep(5)
        
        df_inst = pd.DataFrame(instructions)
        df_dur = pd.DataFrame(duration)
        df_act = pd.DataFrame(actions)

    app.kill_perf_stat()
    global all_values_abs, all_values
    
    all_values["ipc"] = all_values["instructions"]
    all_values_abs["ipc"] = all_values_abs["instructions"]
    
    all_values_abs = all_values_abs.fillna(0)
    all_values = all_values.fillna(0)
    
    all_values.to_csv("all_values_"+str(mix_num)+".csv")
    all_values_abs.to_csv("all_values_abs_"+str(mix_num)+".csv")
    
    file1 = open(name, "a")  # append mode
    file1.write("\nPearson with delta\n\n")
    print("Delat Pearson", all_values.info())
    for e in evensts:
        if(e != "instructions" and e!= "cycles"):
            c = all_values[e].corr(all_values['ipc'])
            print(c)
            file1.write(e+str(":"+str(c))+"\n")
    
    print("All Pearson", all_values_abs.info())
    file1.write("\nPearson with abs\n")
    for e in evensts:
        if(e != "instructions" and e!= "cycles"):
            c = all_values_abs[e].corr(all_values_abs['ipc'])
            print(c)
            file1.write(e+str(":"+str(c))+"\n")

    file1.close()
    
def read_counters():
    f = open("list1.txt", "r")
    all_events = []
    while True:
        file_line = f.readline()
        file_line = file_line.strip()
        if not file_line:
            break
        else:
            all_events.append(file_line)
    f.close()
    chunks = [all_events[x:x+features] for x in range(0, len(all_events), features)]
    return chunks
    
def main():
    chunks = read_counters()
    for i in range(0, len(chunks), 1):
        global all_values, first_done, last_values, all_values_abs, evensts
        evensts = chunks[i]
        evensts.append("instructions")
        evensts.append("cycles")
        
        all_values = pd.DataFrame(columns=evensts)                   
        all_values_abs = pd.DataFrame(columns=evensts)                   
        last_values = np.zeros(shape=(num_cpu, features+2))
        first_done = 0
        name = "corr"+str(i)+".txt"
        file1 = open(name, "w")
        L = ["This is mix"+str(i)]
        file1.writelines(L)
        file1.close()

        run_app(i, name)

    return

if __name__ == '__main__':
    main()
