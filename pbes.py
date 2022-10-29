import random
import os
import subprocess
import time


class PEBS():
    def __init__(self, num_cpu):
        self.num_cpu = num_cpu
        self.event_list = ["L1-dcache-load-misses",
                           "LLC-load-misses",
                           "l2_rqsts.all_pf",
                           "l2_rqsts.miss",
                           "mem_load_retired.l1_hit",
                           "mem_load_retired.l2_hit",
                           "mem_load_retired.l3_hit",
                           "offcore_requests.all_data_rd",
                           "offcore_requests_buffer.sq_full",
                           "instructions",
                           "cycles"]
        self.maxes = []
        self.mins = []
        for i in range(self.num_cpu):
            for e in range(len(self.event_list)):
                self.maxes.append(0)
                self.mins.append(999999999)

    def make_cmd(self):
        cmd = "sudo perf stat -A -C "
        for cpu in range(self.num_cpu):
            cmd += str(cpu)+","
        cmd += " -e "

        for e in self.event_list:
            cmd += e+","
        cmd = cmd[:-1]
        cmd += " sleep 1"
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
            stats[(l[0], l[2])] = l[1]
        return stats

    def state(self):
        state_p = []
        stats = self.run_perf_stat()
        idx = 0
        vals = []
        insts = []
        for cpu in range(self.num_cpu):
            for e in self.event_list:
                val = int(stats[("CPU"+str(cpu), e)])
                vals.append(val)
                if (val >= self.maxes[idx]):
                    self.maxes[idx] = val
                if (val < self.mins[idx]):
                    self.mins[idx] = val

                ratio = 255 * \
                    ((val - self.mins[idx]) / self.maxes[idx])

                state_p.append(int(ratio))
                idx += 1
                if (e == "instructions"):
                    insts.append(val)

        return state_p, insts

    def print(self, stats):
        for cpu in range(self.num_cpu):
            for e in self.event_list:
                print("CPU ", cpu, " Event ", e,
                      " Value ", stats[("CPU"+str(cpu), e)])
                break
