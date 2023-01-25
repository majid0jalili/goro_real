import random
import os
import subprocess
import time
import collections 

class Prefetcher():
    def __init__(self, num_cpu, num_pf_per_core) -> None:
        self.num_cpu = num_cpu
        self.num_pf_per_core = num_pf_per_core
        self.last_setting = []
        
        for core in range(self.num_cpu):
            setting = []
            for pf in range(num_pf_per_core):
                setting.append(0)
            self.last_setting.append(setting)
        self.all_prefetcher_set_forced()
        print("self.last_setting", self.last_setting)

    def prefetcher_set(self, core, mask):
        int_mask = int("".join(str(i) for i in mask), 2)

        if(int_mask == 0):
            cmd = "sudo wrmsr 0xc0011022 -p "+str(core)+" 0xc000000401500000"
        else:
            cmd = "sudo wrmsr 0xc0011022 -p "+str(core)+" 0xc00000040150A000"
        process = subprocess.Popen(cmd,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   shell=True)
        out, err = process.communicate()


    def all_prefetcher_set_forced(self):
        for core in range(self.num_cpu):
            self.prefetcher_set(2*core, "0")
        
        
        
    def all_prefetcher_set(self, mask):
        acc = []
        for core in range(self.num_cpu):
            if mask[0] != self.last_setting[core]:
                self.prefetcher_set(2*core, mask[core])
            for m in mask[core]:
                acc.append(m)
            self.last_setting[core] = mask[core]
        return acc

    def all_prefetchers_on(self):
        last_setting = []
        for core in range(self.num_cpu):
            setting = []
            for pf in range(self.num_pf_per_core):
                setting.append(0)
            last_setting.append(setting)
        self.all_prefetcher_set(last_setting)
        
    def all_prefetchers_off(self):
        last_setting = []
        for core in range(self.num_cpu):
            setting = []
            for pf in range(self.num_pf_per_core):
                setting.append(1) 
            
            last_setting.append(setting)
        self.all_prefetcher_set(last_setting)


