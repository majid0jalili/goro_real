import random
import os
import subprocess
import time

# sudo perf stat -a  -M DRAM_BW_Use -e  llc_misses.mem_read,L1-dcache-load-misses,l2_rqsts.miss,offcore_requests.all_data_rd,offcore_requests_buffer.sq_full,uops_executed.stall_cycles,instructions
class PEBS():
    def __init__(self, num_cpu):
        self.num_cpu = num_cpu
        self.perf_read = 0
        self.event_list = ["L1-dcache-load-misses", # 92
                           "LLC-load-misses",# 39
                           "LLC-store-misses",# 45
                           "node-load-misses",# 34
                           "dTLB-load-misses",# 25
                           "branch-misses",# 05
                           "instructions"# 1
                           ]
        self.inference = [
            "instructions",
            "cycles"
        ]

        self.maxes = []
        self.mins = []
        for i in range(self.num_cpu):
            for e in range(len(self.event_list)):
                self.maxes.append(0)
                self.mins.append(999999999)

    def make_cmd(self):
        cmd = "sudo perf stat -a -A -C "
        for cpu in range(self.num_cpu):
            cmd += str(2*cpu)+","
        cmd += " -e "

        for e in self.event_list:
            cmd += e+","
        cmd = cmd[:-1]
        cmd += " sleep 0.1"

        return cmd

    def run_perf_stat(self):
        cmd = self.make_cmd()
        # print("**********************Running cmd ", cmd)
        process = subprocess.Popen(cmd,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   shell=True)

        _, err = process.communicate()
        lines = err.decode().split('\n')[3:-4]
        stats = {}
        for l in lines:
            l = " ".join(l.split())
            l = l.split(" ")
            if(l[2] == "Bytes"):
                stats[(l[0], l[3])] = l[1]
            else:
                stats[(l[0], l[2])] = l[1]
                
        return stats

    def state(self):
        state_p = []
      
        stats = self.run_perf_stat()
  
       
        idx = 0
        vals = []
        insts = []
        LLC_store_miss = []
        LLC_load_miss = []

        self.perf_read += 1
        if (self.perf_read == 100):
            self.perf_read = 0
            idx = 0
            for i in range(self.num_cpu):
                for e in range(len(self.event_list)):
                    self.maxes[idx] = 0
                    self.mins[idx] = 999999999
                    idx += 1

        idx = 0
        for cpu in range(self.num_cpu):
            for e in self.event_list:
                val = int(stats[("CPU"+str(cpu*2), e)])
                if(e == "LLC-load-misses"):
                    LLC_load_miss.append(val)
                if(e == "LLC-store-misses"):
                    LLC_store_miss.append(val)
                    
                
                vals.append(val)
                if (val >= self.maxes[idx]):
                    self.maxes[idx] = val
                if (val < self.mins[idx]):
                    self.mins[idx] = val

                ratio = 0
                if (self.maxes[idx]):
                    ratio = 1 * \
                        ((val - self.mins[idx]) / self.maxes[idx])

                # state_p.append(int(ratio))
                state_p.append(ratio)
                idx += 1
                
                if (e == "instructions"):
                    insts.append(val)

        LLC_miss = []
        for i in range(len(LLC_store_miss)):
            LLC_miss.append(LLC_store_miss[i]+LLC_load_miss[i])
        return state_p, insts, LLC_miss

    def stats(self):
        state_p = []
        stats = self.run_perf_stat()

        insts = []

        for cpu in range(self.num_cpu):
            for e in self.event_list:
                if (e == "instructions"):
                    val = int(stats[("CPU"+str(2*cpu), e)])
                    insts.append(val)

        return insts

    def print(self, stats):
        for cpu in range(self.num_cpu):
            for e in self.event_list:
                print("CPU ", cpu, " Event ", e,
                      " Value ", stats[("CPU"+str(cpu), e)])
                break
