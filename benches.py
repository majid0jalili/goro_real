import random
import os
import subprocess
import time
import signal
from subprocess import Popen
import pandas as pd
import psutil

spec_root = "/home/cc/spec/benchspec/CPU/"
spec_path = {
    "mcf": "505.mcf_r/run/run_base_refrate_mytest-m64.0000/",
    "lbm": "519.lbm_r/run/run_base_refrate_mytest-m64.0000/",
    # "gcc": "502.gcc_r/run/run_base_refrate_mytest-m64.0000/",
    # "sleep": "./",
    
    # "omnet": "520.omnetpp_r/run/run_base_refrate_mytest-m64.0000/",
    # "fotonik": "549.fotonik3d_r/run/run_base_refrate_mytest-m64.0000/",
    # "pr": "gapbs/",
    # "sssp": "gapbs/",
    "bc": "gapbs/"
}

spec_cmds = {
    "mcf": "./mcf_r_base.mytest-m64 ./inp.in",
    "lbm": "./lbm_r_base.mytest-m64 3000 reference.dat 0 0 100_100_130_ldc.of",
    # "gcc": "./cpugcc_r_base.mytest-m64 gcc-pp.c -O3 -finline-limit=0 -fif-conversion -fif-conversion2 -o gcc-pp.opts-O3_-finline-limit_0_-fif-conversion_-fif-conversion2.s",
    # "sleep": "sleep 10m",
    # "omnet": "./omnetpp_r_base.mytest-m64 -c General -r 0",
    # "fotonik": "./fotonik3d_r_base.mytest-m64",
    # "pr": "/home/cc/gapbs/pr -u 20 -n 10",
    # "sssp": "/home/cc/gapbs/sssp -u 23 -n 20",
    "bc": "/home/cc/gapbs/bc -u 23 -n 20",
}


class Applications():
    def __init__(self, num_app):
        self.blocked = 0
        self.num_app = num_app
        self.core_PID = {}
        self.start_time = {}
        self.end_time = {}
        self.bw_PID = -2
        print("---------------")

        for i in range(self.num_app):
            self.core_PID[2*i] = -2
            self.start_time[2*i] = -2
            self.end_time[2*i] = -2

        print("App map is ", self.core_PID)

    def app_rest(self):
        self.blocked = 0
        self.core_PID = {}
        self.start_time = {}
        self.end_time = {}
        self.bw_PID = -2
        print("---------------")

        for i in range(self.num_app):
            self.core_PID[2*i] = -2
            self.start_time[2*i] = -2
            self.end_time[2*i] = -2

        print("App map is ", self.core_PID)
    
    def kill_bw(self):
        cmd_bg = "sudo pkill -f pcm-memory*"
        print("run_bw killed the last run the new", cmd_bg, self.bw_PID)
        process = subprocess.Popen(cmd_bg,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               shell=True)
        output, errors = process.communicate()
        print(output, errors)
    
    def kill_perf_stat(self):
        cmd_bg = "sudo pkill -f perf*"
        print("kill_perf_stat killed the last run the new", cmd_bg, self.bw_PID)
        process = subprocess.Popen(cmd_bg,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               shell=True)
        output, errors = process.communicate()
        print(output, errors)
        
    def run_bw(self, output):
        self.kill_bw()
        cmd_path = "./"
        cmd_bg = "sudo /home/cc/test/pcm/build/bin/pcm-memory -csv="+str(output)+".csv"
        process = subprocess.Popen(cmd_bg,
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL,
                                   shell=True,
                                   cwd=cmd_path
                                   )
        print("-**----run_bw----PID", process.pid, self.bw_PID)

   
    def run_perf_stat(self):
        self.kill_perf_stat()
        cmd_path = "./"
        cmd_bg = "sudo perf stat -x, -C 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30 -o test.csv -A -I 100 -e branches,cache-references,instructions,L1-dcache-load-misses,L1-dcache-loads,L1-dcache-prefetches,L1-icache-load-misses,dTLB-load-misses,dTLB-loads,iTLB-loads,msr/aperf/,msr/irperf/,msr/mperf/,msr/tsc/,branch-instructions,branch-misses,branch-loads  --append "
        process = subprocess.Popen(cmd_bg,
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL,
                                   shell=True,
                                   cwd=cmd_path
                                   )
        print("-**---run_perf_stat-----PID", process.pid)
        
    def print_apps(self):
        print("Apps running")
        print(self.core_PID)
        print(self.start_time)
    
    def duration(self):
        dur = []
        for i in range(self.num_app):
            idx = i*2
            dur.append(self.end_time[idx] - self.start_time[idx])
        return dur
        
    def force_kill_all(self):
        for i in range(self.num_app):
            os.kill(self.core_PID[i], 9)
            self.core_PID[2*i] = -2
            self.start_time[2*i] = -2
            self.end_time[2*i] = -2

    def check_pid(self, pid):
        """ Check For the existence of a unix pid. """
        if(pid < 0):
            return False
        try:
            p = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return False
            
        if p.status() == psutil.STATUS_ZOMBIE:
            time.sleep(0.5)
            return False
        try:
            os.kill(pid, 0)
            # os.wait()
        except OSError:
            return False
        else:
            return True

    def check_pids(self):
        empty_core = 0
        for core in self.core_PID:
            if (self.check_pid(self.core_PID[core]) == False):
                empty_core += 1
                self.core_PID[core] = -2
                if self.end_time[core] == -2 :
                    self.end_time[core] =  time.time()

        return empty_core

    def num_running_apps(self):
        return self.num_app - self.check_pids()

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
            return
        core = self.find_first_empty_core()
        if core == -1:
            return
        cmd_path, cmd_bg = self.get_spec_app(core)
        print("Running ", cmd_bg, " on core ", core)
        process = subprocess.Popen(cmd_bg,
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL,
                                   shell=True,
                                   cwd=cmd_path
                                   )
        self.core_PID[core] = process.pid
        self.start_time[core] = time.time()
        self.end_time[core] = -2

        return cmd_path, cmd_bg

    def replay_runs(self, cmd_path, cmd_bg):
        idx = 0
        for cmd in cmd_bg:
            print("Running ", cmd, " on core ", idx*2)
            process = subprocess.Popen(cmd,
                                       stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL,
                                       shell=True,
                                       cwd=cmd_path[idx]
                                       )
            self.core_PID[idx*2] = process.pid
            self.start_time[idx*2] = time.time()
            self.end_time[idx*2] = -2
            idx += 1

    def get_sleep_time(self, time):
        cmd = "sleep "+str(time)
        return "./", cmd

    def run_app_min(self, waitime):
        cmd_bg = ""
        cmd_path, cmd_bg1 = self.get_spec_app(0)
        cmd_bg += cmd_bg1

        print("Running ", cmd_bg, " for ", waitime)
        run = True
        start = time.time()
        sofar = 0

        pid = -2
        duration = waitime
        outs, errs = 0, 0

        while (run):
            if (self.check_pid(pid) == False):
                process = subprocess.Popen(cmd_bg,
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE,
                                           shell=True,
                                           cwd=cmd_path
                                           )
                pid = process.pid
                print("launched ", pid)

            try:

                outs, errs = None, None
                print("Waiting for ", duration, outs, errs)
                outs, errs = process.communicate(timeout=duration)
                sofar = time.time()
                elapsed = sofar-start
                duration -= elapsed

                if (elapsed > waitime):
                    print("Done ", elapsed)
                    run = False
                else:
                    print("Need a new launch ", elapsed, waitime-elapsed)

            except subprocess.TimeoutExpired as e:
                print("Timeout expired", process.pid, waitime-elapsed)
                run = False

            finally:
                if errs != None:
                    print("errs ", errs.decode())


# A = Applications(4)
# A.run_perf_stat()
# time.sleep(10)
# A.kill_perf_stat()