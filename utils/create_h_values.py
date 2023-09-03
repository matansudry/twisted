import sys
sys.path.append(".")

from utils.general_utils import convert_topology_to_str, load_pickle, save_pickle
import os
import tqdm

def main():
    paths = ["datasets/21_links/4_new/"]
    h_values = {}
    for path in paths:
        files = os.listdir(path)
        for file in tqdm.tqdm(files):
            if os.path.isdir(path+file):
                continue
            item = load_pickle(path+file)
            str_topology = convert_topology_to_str(item["topology_end"])
            if str_topology in h_values.keys():
                h_values[str_topology] += 1
            else:
                h_values[str_topology] = 1
    
    save_pickle("h_values_4_crosses.txt", h_values)
    temp=1

if __name__ == '__main__':
    main()