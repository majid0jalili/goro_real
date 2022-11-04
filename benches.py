import random
import os
import subprocess
import time


spec_root = "/home/cc/spec/benchspec/CPU/"
spec_path = {
    "mcf": "505.mcf_r/run/run_base_refrate_mytest-m64.0000/",
    "lbm": "519.lbm_r/run/run_base_refrate_mytest-m64.0000/",
    "gcc": "502.gcc_r/run/run_base_refrate_mytest-m64.0000/",
    "omnet": "520.omnetpp_r/run/run_base_refrate_mytest-m64.0000/",
    "fotonik": "549.fotonik3d_r/run/run_base_refrate_mytest-m64.0000/",
    "pr": "gapbs/",
    "sssp": "gapbs/",
    "bc": "gapbs/"
}

spec_cmds = {
    "mcf": "./mcf_r_base.mytest-m64 ./inp.in",
    "lbm": "./lbm_r_base.mytest-m64 3000 reference.dat 0 0 100_100_130_ldc.of",
    "gcc": "./cpugcc_r_base.mytest-m64 gcc-pp.c -O3 -finline-limit=0 -fif-conversion -fif-conversion2 -o gcc-pp.opts-O3_-finline-limit_0_-fif-conversion_-fif-conversion2.s",
    "omnet": "./omnetpp_r_base.mytest-m64 -c General -r 0",
    "fotonik": "./fotonik3d_r_base.mytest-m64",
    "pr": "/home/cc/gapbs/pr -u 23 -n 20",
    "sssp": "/home/cc/gapbs/sssp -u 23 -n 20",
    "bc": "/home/cc/gapbs/bc -u 23 -n 20",
}


class Applications():
    def __init__(self, num_app):
        self.blocked = 0
        self.num_app = num_app
        self.core_PID = {}

        for i in range(self.num_app):
            self.core_PID[2*i] = -2

        print("App map is ", self.core_PID)

    def force_kill_all(self):
        for i in range(self.num_app):
            os.kill(self.core_PID[i], 9)
            self.core_PID[i] = -2

    def check_pid(self, pid):
        """ Check For the existence of a unix pid. """
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        else:
            return True

    def check_pids(self):
        empty_core = 0
        for core in self.core_PID:
            if (self.check_pid(self.core_PID[core]) == False):
                # print("PID ", self.core_PID[core], " is not running")
                empty_core += 1
                self.core_PID[core] = -2

        return empty_core

    def find_first_empty_core(self):
        for core in self.core_PID:
            if self.core_PID[core] == -2:
                return core
        return -1

    def get_spec_app(self, core):
        spec_app = random.choice(list(spec_cmds.keys()))
        cmd_path = spec_root+spec_path[spec_app]
        cmd_bg = "taskset -c "+str(core)+" "+spec_cmds[spec_app]
        return cmd_path, cmd_bg

    def run_app(self):
        empty_core = self.check_pids()
        if empty_core == 0:
            self.blocked += 1
            #print("All cores are busy self.blocked", self.blocked)
            return
        core = self.find_first_empty_core()
        if core == -1:
            # print("All cores are busy")
            return
        cmd_path, cmd_bg = self.get_spec_app(core)
        # print("Running path: ", cmd_path, " cmd: ", cmd_bg)

        process = subprocess.Popen(cmd_bg,
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL,
                                   shell=True,
                                   cwd=cmd_path
                                   )
        self.core_PID[core] = process.pid
        # print("App PID is ", self.core_PID)
