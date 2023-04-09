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
    # "mcf": "505.mcf_r/run/run_base_refrate_mytest-m64.0000/",
    # "lbm": "519.lbm_r/run/run_base_refrate_mytest-m64.0000/",
    # "gcc": "502.gcc_r/run/run_base_refrate_mytest-m64.0000/",
    # "omnet": "520.omnetpp_r/run/run_base_refrate_mytest-m64.0000/",
    # "fotonik": "549.fotonik3d_r/run/run_base_refrate_mytest-m64.0000/",
    
    # "sleep": "./",
       
    "pr": "gapbs/",
    "sssp": "gapbs/",
    "bc": "gapbs/",
    "bfs": "gapbs/",
    "tc": "gapbs/",
    "cc": "gapbs/"
    
    
}

spec_cmds = {
    # "mcf": "./mcf_r_base.mytest-m64 ./inp.in",
    # "lbm": "./lbm_r_base.mytest-m64 3000 reference.dat 0 0 100_100_130_ldc.of",
    # "gcc": "./cpugcc_r_base.mytest-m64 gcc-pp.c -O3 -finline-limit=0 -fif-conversion -fif-conversion2 -o gcc-pp.opts-O3_-finline-limit_0_-fif-conversion_-fif-conversion2.s",
    # "omnet": "./omnetpp_r_base.mytest-m64 -c General -r 0",
    # "fotonik": "./fotonik3d_r_base.mytest-m64",
    
    # "sleep": "sleep 10m",
    
    "pr": "/home/cc/gapbs/pr -u 21 -n 100",
    "sssp": "/home/cc/gapbs/sssp -u 22 -n 20",
    "bc": "/home/cc/gapbs/bc -u 21 -n 100",
    "bfs": "/home/cc/gapbs/bfs -u 22 -n 100",
    "tc": "/home/cc/gapbs/tc -u 22 -n 20",
    "cc": "/home/cc/gapbs/cc -u 22 -n 100"
    
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
            self.core_PID[i] = -2
            self.start_time[i] = -2
            self.end_time[i] = -2

        print("App map is ", self.core_PID)

    def app_rest(self):
        self.blocked = 0
        self.core_PID = {}
        self.start_time = {}
        self.end_time = {}
        self.bw_PID = -2
        print("---------------") 

        for i in range(self.num_app):
            self.core_PID[i] = -2
            self.start_time[i] = -2
            self.end_time[i] = -2

        print("App map is ", self.core_PID)
    
    def add_noise(self):
        cmd_bg = "taskset -c 128-191:1 /home/cc/test/STREAM/stream"
        cmd_path = "./"
        process = subprocess.Popen(cmd_bg,
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL,
                                   shell=True,
                                   cwd=cmd_path
                                   )
        print("Noise added ", process.pid)
    
    def kill_bw(self):
        cmd_bg = "sudo pkill -f AMDuProfPcm*"
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
        cmd_bg = "sudo /opt/AMDuProf_4.0-341/bin/AMDuProfPcm -r -m memory -d 10000 -A system -o "+str(output)+".csv"
        print("cmd_bg", cmd_bg)
        process = subprocess.Popen(cmd_bg,
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL,
                                   shell=True,
                                   cwd=cmd_path
                                   )
        print("-**----run_bw----PID", process.pid, self.bw_PID)

   
    def run_perf_stat(self, event_list):
        self.kill_perf_stat()
        cmd_path = "./"
        
        cmd_bg = "sudo perf stat -x, -C 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63 -o test.csv -A -I 100 -e "+event_list+  " --append "
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
            idx = i
            dur.append(self.end_time[idx] - self.start_time[idx])
        return dur
        
    def force_kill_all(self):
        for i in range(self.num_app):
            os.kill(self.core_PID[i], 9)
            self.core_PID[i] = -2
            self.start_time[i] = -2
            self.end_time[i] = -2

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
            print("Replaying ", cmd, " on core ", idx)
            process = subprocess.Popen(cmd,
                                       stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL,
                                       shell=True,
                                       cwd=cmd_path[idx]
                                       )
            self.core_PID[idx] = process.pid
            self.start_time[idx] = time.time()
            self.end_time[idx] = -2
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

# num_cpu=64
# app = Applications(num_cpu)
# cmds = []
# paths = []
    
# for i in range(num_cpu):
    # cmd_path, cmd_bg = app.get_spec_app(i*2)
    # cmds.append(cmd_bg)
    # paths.append(cmd_path)

# app.replay_runs(paths, cmds)
# A.run_bw("aaa")
# time.sleep(5)
# A.kill_bw()

# A.run_perf_stat()
# time.sleep(10)
# A.kill_perf_stat()