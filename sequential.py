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
parser.add_argument('--lr_rate', type=float, default=1e-6,
                    help='learning rate (default : 0.0001)')
parser.add_argument('--batch_size', type=int, default=64,
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
num_features_per_core = 257

state_space = num_cpu*num_features_per_core
action_space = num_cpu
action_scale = pow(2, num_pf_per_core)

alpha = 0.2
beta = 0.6


device = 'cuda' if torch.cuda.is_available() else 'cpu'


events = []

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

def run_model(agent):
        
    tot_actions = [0] * num_cpu
    for i in range(100):  
        state = torch.rand((20, state_space)).float().to(device)
        action = agent.action_test(state)
        for a in action:
            tot_actions[a]+=1
    tot = 0
    for a in tot_actions:
        tot += a
    for a in range(len(tot_actions)):
        tot_actions[a] = tot_actions[a]/(tot*1.0)
        
    return tot_actions
        
def train(agent):
    global events
    print("Function set_collector")
    event_list = ""
    for e in events:
        event_list +=e+","
    event_list = event_list[:-1]
    
    app = Applications(num_cpu)
    app.run_perf_stat(event_list)
    time.sleep(4)
    
    pebs = PEBS(num_cpu)
    pf = Prefetcher(num_cpu, num_pf_per_core)
    
    total_reward = 0
    n = 0
    state, insts, llc_misses = pebs.state1(events)
    
    
    
    next_state = []
    next_inst = []

    avg_reward = 0
    total_llc_misses = llc_misses
    total_insts = [0]*len(insts)
    avg_inst = insts
    
    itr = 0
    examine = 0
    loss = 0
    elapsed = {} 
    time.sleep(1)
    
    
    print("Start--------------------------------------")
    print("len(state)", len(state))
    while (True):
        for i in range(num_cpu):
            app.run_app()
        
        
        tic = time.time()
        action = agent.action(state, False)
        toc = time.time()
        elapsed["action"]=round(toc-tic, 2)
        
        tic = time.time()
        action = pf.all_prefetcher_set(action)
        toc = time.time()
        elapsed["pf_Set"]=round(toc-tic, 2)
        
        tic = time.time()
        next_state, next_inst, llc_misses = pebs.state1(events)
        toc = time.time()
        elapsed["read_state"]=round(toc-tic, 2)

        # time.sleep(0.1)

        total_insts = [sum(i) for i in zip(total_insts, next_inst )]  
        itr += 1
        examine += 1
        
        reward = 0
        for inst in range(len(insts)):
            if (insts[inst] != 0):
                inst_ratio = (next_inst[inst] / avg_inst[inst]) - 1
                if(inst_ratio > 4):
                    inst_ratio = 4
                if(inst_ratio < -4):
                    inst_ratio = -4
                
                # if(inst < (num_cpu/2)):
                    # if(inst_ratio < 0):
                        # reward -= 1
                
                        
                reward += 1.0*inst_ratio
                 

        r_arr = [reward]
        avg_reward += reward
        
        with open(r'./reward.txt', 'a') as fp:
            fp.write("examine "+ str(examine)+" ")
            fp.write('reward ' + str(round(reward, 3)) )
                     # ' loss '+str(loss)+" Misses ")
            # fp.write(" ".join(str(item) for item in total_llc_misses))
            fp.write(", Instructions ")
            fp.write(" ".join(str(item) for item in next_inst))
            fp.write(", Avg ")
            fp.write(" ".join(str(item) for item in avg_inst))
            
            fp.write("\n")
            fp.close()
        
        
                
        if (itr == 1):
            
            avg_reward /= 1
            for inst in range(len(insts)):
                avg_inst[inst] = total_insts[inst]/itr

            total_insts = [0]*len(total_insts)
            itr = 0
            avg_reward = 0
        
        if examine == 100:
            print(elapsed)
            examine = 0
            agent.save_model("model")
            agent.memory.beta = beta
            # agent.memory.write_to_csv("mem.xlsx")
            dist_actions = run_model(agent)
            print("dist_actions", dist_actions)
            # if(dist_actions[0]+dist_actions[1] > 0.8):
            if(dist_actions[0] > 0.8):
                print("Very close to Intel")
                break
            
            
        agent.memory.write_buffer(state, next_state, action, r_arr)

        if (agent.memory.size() > 2*batch_size):
            tic = time.time()
            loss = agent.train_model(batch_size, gamma)
            agent.memory.beta = beta + (itr/100)*(1-beta)
            toc = time.time()
            elapsed["train"]=(toc-tic)
        else:
            time.sleep(0.4)
       
   
        state = next_state
        insts = next_inst

def read_counters():
    f = open("ag.list", "r")
    all_events = []
    while True:
        file_line = f.readline()
        file_line = file_line.strip()
        if not file_line:
            break
        else:
            all_events.append(file_line)
    f.close()
    all_events.append("instructions")
    all_events.append("cycles")
    print("all_events size", len(all_events))
    return  all_events  


def load_model(agent):
    print("Function load_model")
    agent.load_model("./models/model", device)


def main():
    global events
    events = read_counters()

    agent = BQN(state_space, action_space, action_scale,
            learning_rate, device, num_cpu, num_pf_per_core, alpha, beta)
      
    # agent.load_model("./models/model", device)
    
    train(agent)
if __name__ == '__main__':
    main()
    