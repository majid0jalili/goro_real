import re
import os
import argparse
import multiprocessing
import time
import csv
import pandas as pd
from random import random
from random import randint
from datetime import datetime

from benches import Applications
from prefetcher import Prefetcher


num_cpu = 16
num_pf_per_core = 4

def make_action(run_type):
    action_start = time.time()
    actions = []
    
    if(run_type == "base"):
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
    time.sleep(0.1)
    start = time.time()
    state, insts, llc_misses = pebs.state1()
    end = time.time()
    return state, insts, round(end-start, 2)


def make_app(num_actual_app):
    cmds = []
    paths = []
    app = Applications(num_cpu)
    for i in range(num_actual_app):
        cmd_path, cmd_bg = app.get_spec_app(i*2)
        cmds.append(cmd_bg)
        paths.append(cmd_path)
    return cmds, paths

def run_app(mix_num, run_mode, cmds, paths):
    app = Applications(num_cpu)
    pf = Prefetcher(num_cpu, num_pf_per_core)
    pf.all_prefetcher_set_forced()
    app.replay_runs(paths, cmds)

    itr = 0
    app.run_bw("app_"+str(mix_num)+str("_bw_"+str(run_mode)))
    while(app.num_running_apps() != 0):
        action, make_action_length = make_action(run_mode)
        take_action_length = take_action(action, pf)
        time.sleep(0.2)
        if(itr == 100):
            itr = 0
            print("{} take_action:{} make_action:{} ".format(run_mode, take_action_length, make_action_length)) 
            print(app.core_PID)
        itr += 1
        
    duration = app.duration()
    app.kill_bw()
    return duration
    





def main():

    for i in range(16, 18, 2):
        cmds, paths = make_app(i)
        run_app(i, "base", cmds, paths)
        run_app(i, "random", cmds, paths)

    return

if __name__ == '__main__':
    main()
