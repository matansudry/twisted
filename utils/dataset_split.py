import sys
sys.path.append(".")

import pickle
from sklearn.model_selection import train_test_split
import os
import tqdm
import random
import shutil

def change_name(path, old_name, name_name):
    os.rename(path+old_name, path+name_name)
    return name_name

def main():
    path = "datasets/21_links/3/"
    input_path = path#+"1/"
    output_train = "train/"
    output_val = "test/"
    ratio = 0.05
    files = os.listdir(input_path)
    for file in tqdm.tqdm(files):
        if os.path.isdir(input_path+file):
            continue
        
        #with open(path+file, "rb") as fp:   # Unpickling
            #dataset_dict = pickle.load(fp)
        random_number = random.uniform(0, 1)
        #if "batch_2_" not in file:
            #file = change_name(input_path, file, "batch_2_"+file)
        if random_number < ratio:
            shutil.move(input_path+file, path+output_val+file)
            #with open(path+output_val+file, "wb") as fp:
                #pickle.dump(dataset_dict, fp)
        else:
            shutil.move(input_path+file, path+output_train+file)
            #with open(path+output_train+file, "wb") as fp:
                #pickle.dump(dataset_dict, fp)
if __name__ == '__main__':
    main()
