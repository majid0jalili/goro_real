import json

def read(file):
# Open the JSON file
    with open(file, 'r') as f:
        data = json.load(f)

    for e in data:
        if("UMask" in e):
            print(str("r")+e["UMask"][2:]+e["EventCode"][2:])



read("cache.json")
read("data.json")
read("fp.json")
read("other.json")
read("rec.json")
read("core.json")
read("mem.json")
read("branch.json")
