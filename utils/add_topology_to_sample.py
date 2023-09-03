import sys
sys.path.append(".")

import tqdm 
from utils.topology.state_2_topology import state2topology
import torch
import pickle
import os

def main():
    paths = ["datasets/new_train/"]
    new_paths = ["datasets/new_train_with_topology/"]
    for index, path in enumerate(paths):
        files = os.listdir(path)
        for file in tqdm.tqdm(files):
            with open(path+file, "rb") as fp:   # Unpickling
                try:
                    dataset_dict = pickle.load(fp)
                except:
                    print(file)
                    raise
            if isinstance(dataset_dict, dict):
                for item in dataset_dict.values():
                    current_state = item["end"]
                    topology = state2topology(torch.tensor(current_state))
                    topology = topology[0].points
                    item["topology_end"] = topology
                    
                with open(new_paths[index]+file, "wb") as fp:
                    pickle.dump(dataset_dict, fp)

if __name__ == '__main__':
    main()




