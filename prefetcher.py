import random
import os
import subprocess
import time


class Prefetcher():
    def __init__(self, num_cpu) -> None:
        self.num_cpu = num_cpu

    def prefetcher_set(self, core, mask):
        int_mask = int("".join(str(i) for i in mask), 2)
        cmd = "sudo wrmsr 0x1a4 -p "+str(core)+" "+str(int_mask)
        #print("Running cmd ", cmd)
        process = subprocess.Popen(cmd,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   shell=True)
        out, err = process.communicate()

    def all_prefetcher_set(self, mask):
        acc = []
        for core in range(self.num_cpu):
            self.prefetcher_set(core, mask[core])
            for m in mask[core]:
                acc.append(m)
        return acc
