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

    def prefetcher_set(self, core, mask):
        int_mask = int("".join(str(i) for i in mask), 2)
        print("int_mask", int_mask)
        cmd = "sudo wrmsr 0x1a4 -p "+str(core)+" "+str(int_mask)
        process = subprocess.Popen(cmd,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   shell=True)
        out, err = process.communicate()
        print(out)

    def all_prefetcher_set_forced(self):
        for core in range(self.num_cpu):
            self.prefetcher_set(2*core, "0")
        
        
        
    def all_prefetcher_set(self, mask):
        acc = []
        for core in range(self.num_cpu):
            if collections.Counter(mask[core]) != collections.Counter(self.last_setting[core]):
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
        self.all_prefetcher_set(self.last_setting)

