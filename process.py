import pandas as pd
import glob

path = "./"

isolation = {
"pr": 34,
"bc": 162,
"lbm": 167,
"mcf": 283
}

apps = ["lbm", "mcf", "pr", "bc"]

def maxavg(path, filename, all_time):
    xl = pd.ExcelFile(path + filename)
    time_sheets = ["duration_base", "duration_goro", "duration_random"]

    mean = []
    mmax = []
    sums = []

    
    for sheet in time_sheets:
        df = xl.parse(sheet)
        df["iso"] = all_time
        df["w"] = df["iso"]/df[0]
        
        df.to_csv("a.csv")
        # mean.append(df[0].mean(axis=0).to_string(index=False))
        mmax.append(df[0].max(axis=0))
        sums.append(df["w"].sum(axis=0))

    # print(filename[2:-5], str(mean)[1:-1], str(mmax)[1:-1])
    print(filename[2:-5], str(mmax)[1:-1], "Sums ", str(sums)[1:-1])
    
def find_order(path, filename):
    all_time = []
    xl = pd.ExcelFile(path + filename)
    df = xl.parse("apps")
    for i, j in df.iterrows():
        for app in apps:
            if(app  in str(j)):
                all_time.append(isolation[app])
                break
    return all_time

def list_files(path):
    allpath = path+'app_*.xlsx'
    files = glob.glob(allpath)
    return files

def check_all(path, all_time):
    # print(isolation)
    files = list_files(path)
    print("Files ", len(files))
    for f in files:
        maxavg(path, f, all_time)
        

# 

all_time  = find_order(path, "app_NoNoise_74.xlsx")
check_all(path, all_time)