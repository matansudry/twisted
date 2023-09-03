import sys
sys.path.append(".")
import pickle

if __name__ == '__main__':
    path = "system_flow/logs/6_10--21:29:13/log.pkl"
    with open(path, "rb") as fp:   # Unpickling
        log = pickle.load(fp)
    temp=1
