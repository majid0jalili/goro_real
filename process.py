import pandas as pd
import glob

path = "./"

def maxavg(path, filename):
    xl = pd.ExcelFile(path + filename)
    time_sheets = ["duration_base", "duration_goro", "duration_random"]

    mean = []
    mmax = []
    sheets = []
    
    for sheet in time_sheets:
        df = xl.parse(sheet)

        sheets.append(sheet)
        mean.append(df.mean(axis=0).to_string(index=False))
        mmax.append(df.max(axis=0).to_string(index=False))

    print(filename[2:-5], str(mean)[1:-1], str(mmax)[1:-1])
    

def list_files(path):
    allpath = path+'app_*.xlsx'
    files = glob.glob(allpath)
    return files

def check_all(path):
    files = list_files(path)
    for f in files:
        maxavg(path, f)
        

check_all(path)
