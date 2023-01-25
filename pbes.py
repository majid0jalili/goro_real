import random
import os
import subprocess
import time
import pandas as pd

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

    def state1(self):
        # print("Reading...")
        try:
            df = pd.read_csv('test.csv', skiprows=2, header=None, on_bad_lines='skip')
            # print(df)
            # print(df.shape)
        except Exception as e:
            print("Exception gotta wait for 1 sec")
            time.sleep(1)
            print("read again")
            df = pd.read_csv('test.csv', skiprows=2, header=None, on_bad_lines='skip')
            
        df_empty = pd.DataFrame() 
        df_empty.to_csv('test.csv', index=False)

        
        if(df.shape[1] == 9):
            df.columns = ['A', 'CPUs', 'Values', 'D', 'Events', 'F', 'G', 'H', 'I']
            df = df.drop(columns=['A', 'D', 'F', 'G', 'H', 'I'])
        elif(df.shape[1] == 11):
            df.columns = ['A', 'CPUs', 'Values', 'D', 'Events', 'F', 'G', 'H', 'I', "J", "K"]
            df = df.drop(columns=['A', 'D', 'F', 'G', 'H', 'I', "J", "K"])
        
        df.Values =pd.to_numeric(df.Values, errors ='coerce').fillna(0).astype('int')
        df1 = df.groupby(['CPUs', 'Events']).mean(numeric_only=True)
        # df1.to_csv("groupby.csv")
        features = {}
        try:
            features = df1["Values"].to_dict()
        except Exception as e:
            print("Exception happned here", e)
            print(df)
            print("-----")
            print(df1)
            print("-----")

# ,,,,,,  

        features_clean = {}
        evensts = ["branches", #A
                   "cache-references", #B
                   "instructions",  #C
                   "L1-dcache-load-misses", #D
                   "L1-dcache-loads", #E
                   "L1-dcache-prefetches",  #F
                   "L1-icache-load-misses",  #G
                   "dTLB-load-misses", #H
                   "dTLB-loads", #I
                   "iTLB-loads ", #J
                   
                   "msr/aperf/", #K
                   "msr/irperf/",#L
                   "msr/mperf/",
                   "msr/tsc/",
                   "branch-instructions",
                   "branch-misses",
                   "branch-loads"
                   
                   ] 

        for cpu in range(0, self.num_cpu*2, 2):
            for e in evensts:
                thekey = ("CPU"+str(cpu), e)
                if thekey in features:
                    features_clean[thekey] = features[thekey]
                else:
                    features_clean[thekey] = 1
                    
       
        features_list = []
        insts = []

        for f in features_clean:
            cpu = f[0][3:]
            if (f[1] != 'instructions'):
                features_list.append(features_clean[f] /
                                     features_clean[("CPU" + str(cpu),
                                              "instructions")]
                                     )
            else:
                insts.append(features_clean[f])
    

        # print(features_list)
        # print("SIZE---", len(features_list))
        return features_list, insts, []
    
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


# P = PEBS(4)
# P.state1()