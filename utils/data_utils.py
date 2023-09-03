import os
import tqdm
import numpy as np
import shutil
import pickle
from utils.general_utils import *
from dm_control import mujoco

def change_files_names():
    input_paths = ["datasets/"]
    new_path = "datasets/stage_1/"
    test_path = "datasets/test/"
    for input_path in input_paths:
        files = os.listdir(input_path)
        cnt = 0
        for file in tqdm.tqdm(files):
            if "txt" in file:
                os.rename(input_path+file, input_path+"new_"+file)

        files = os.listdir(input_path)
        for file in tqdm.tqdm(files):
            if "txt" in file:
                shutil.move(input_path + file, new_path + file)

def concat_data():
    input_path = "datasets/test/"
    files = os.listdir(input_path)
    cnt = 0
    save_index = 0
    full_dataset = []
    for file in tqdm.tqdm(files):
        if cnt >= 100:
            with open(input_path+"stage_3_new_"+str(save_index)+".txt", "wb") as fp:
                pickle.dump(full_dataset, fp)
            save_index += 1
            full_dataset = []
            cnt = 0
        if "stage_3" in file:
            with open(input_path+file, "rb") as fp:   # Unpickling
                dataset = pickle.load(fp)
                for item in dataset.values():
                    full_dataset.append(item)
                    cnt +=1

def convert_qpos_to_xanchor():
    physics = mujoco.Physics.from_xml_path('assets/rope_v3.xml')
    paths = ["datasets/new_train_with_topology/"]
    new_folder = ["datasets/new_train_with_topology/"]
    for path_index, path in enumerate(paths):
        files = os.listdir(path)
        for file in tqdm.tqdm(files):
            with open(path+file, "rb") as fp:   # Unpickling
                try:
                    dataset_dict = pickle.load(fp)
                except:
                    print(file)
                    raise
                
            if isinstance(dataset_dict, list):
                for item in dataset_dict:
                    #start
                    xyz = convert_qpos_to_xyz(physics,item['start_data'])
                    item['start'] = xyz
                    #end
                    xyz = convert_qpos_to_xyz(physics,item['end_data'])
                    item['end'] = xyz
                with open(new_folder[path_index]+file, "wb") as fp:
                    pickle.dump(dataset_dict, fp)

def split_data_to_train_and_test():
    input_paths = ["datasets/new_stage_1/"]
    train_path = "datasets/train/"
    test_path = "datasets/test/"
    for input_path in input_paths:
        files = os.listdir(input_path)
        for file in tqdm.tqdm(files):
            prob = np.random.uniform(low=0.0, high=1.0)
            if prob < 0.1:
                shutil.move(input_path + file, test_path + file)
            else:
                shutil.move(input_path + file, train_path + file)

def fixing_the_data_to_0_to_2_pai(config_length):
    #physics = mujoco.Physics.from_xml_path('assets/rope_v2.xml')
    path = "datasets/test/"
    #output = "datasets/after_fix/"
    files = os.listdir(path)
    for file in tqdm.tqdm(files):
        if "stage_3" not in file:
            continue
        with open(path+file, "rb") as fp:   # Unpickling
            try:
                dataset_dict = pickle.load(fp)
            except:
                print(file)
                raise
            if isinstance(dataset_dict, dict):
                for item in dataset_dict.values():
                    for index, _ in enumerate(item['start_data'][7:config_length]):
                        while item['start_data'][index+7] < -np.pi:
                            item['start_data'][index+7] += np.pi*2
                        while item['start_data'][index+7] > np.pi:
                            item['start_data'][index+7] -= np.pi*2
                    for index, _ in enumerate(item['end_data'][7:config_length]):
                        while item['end_data'][index+7] < -np.pi:
                            item['end_data'][index+7] += np.pi*2
                        while item['end_data'][index+7] > np.pi:
                            item['end_data'][index+7] -= np.pi*2


                with open(path+file, "wb") as fp:
                    pickle.dump(dataset_dict, fp)
            
            elif isinstance(dataset_dict, list):
                for item in dataset_dict:
                    for index, _ in enumerate(item['start_data'][7:config_length]):
                        while item['start_data'][index+7] < -np.pi:
                            item['start_data'][index+7] += np.pi*2
                        while item['start_data'][index+7] > np.pi:
                            item['start_data'][index+7] -= np.pi*2
                    for index, _ in enumerate(item['end_data'][7:config_length]):
                        while item['end_data'][index+7] < -np.pi:
                            item['end_data'][index+7] += np.pi*2
                        while item['end_data'][index+7] > np.pi:
                            item['end_data'][index+7] -= np.pi*2


                with open(path+file, "wb") as fp:
                    pickle.dump(dataset_dict, fp)

if __name__ == '__main__':
    change_files_names()