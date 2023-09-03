import pickle
from re import L
import numpy as np
import os
import tqdm
from torch.utils.data import Dataset
from utils.general_utils import load_pickle

class CustomDataset(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir

    def __len__(self):
        return len(os.listdir(self.dataset_dir))

    def __getitem__(self, idx):
        sample_path = os.path.join(self.dataset_dir, str(idx)+".txt")
        sample = load_pickle(sample_path)
        input = sample[0]
        gt_pos = sample[1]
        gt_qpos = sample[2]
        return input, gt_pos, gt_qpos

class CustomDataset_s2a(CustomDataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir

    def __len__(self):
        return len(os.listdir(self.dataset_dir))

    def __getitem__(self, idx):
        sample_path = os.path.join(self.dataset_dir, str(idx)+".txt")
        sample = load_pickle(sample_path)
        input = sample[0]
        gt_pos = sample[1]
        return input, gt_pos


def convert_action_from_index_to_one_hot_vector(action, num_of_links):
    one_hot = np.zeros(num_of_links+3)
    one_hot[int(action[0])] = 1
    one_hot[num_of_links] = action[1]
    one_hot[num_of_links+1] = action[2]
    one_hot[num_of_links+2] = action[3]

    return one_hot

def fix_yaw_problem(sample, config_length):
    for index in range(len(sample[7:config_length])):
        while sample[index+7] > np.pi:
            sample[index+7] -= np.pi*2
        while sample[index+7] < -np.pi:
            sample[index+7] += np.pi*2
    return sample

def preprocessing_dataset_qpos(cfg, paths, num_of_samples=None, index=None, return_qpos=False, return_with_init_position=False,\
     val_topology=False,low_limit=-1, max_limit=20):
    mapping_old = {
        'start_config': 'start_data',
        'end_config': 'end_data',
        'action': 'action',
        'start_pos': 'start',
        'end_pos': 'end',
        'topology_start': 'topology_start',
        'topology_end': "topology_end",
    }
    mapping_new = {
        'start_config': 'start_config',
        'end_config': 'end_config',
        'action': 'action',
        'start_pos': 'start_pos',
        'end_pos': 'end_pos',
        'topology_start': 'topology_start',
        'topology_end': "topology_end",
    }

    num_of_links = cfg["main"]["num_of_links"]
    config_length = cfg["main"]["config_length"]
    dataset_list = []
    gt_list = []
    if return_qpos:
        qpos_list = []
    topology = []
    for path in paths:
        files = os.listdir(path)
        for file in tqdm.tqdm(files):
            if (num_of_samples is not None and len(dataset_list) > num_of_samples):
                break
            with open(path+file, "rb") as fp:   # Unpickling
                try:
                    dataset_dict = pickle.load(fp)
                except:
                    print(file)
                    raise
                
            if isinstance(dataset_dict, dict):
                for item in dataset_dict.values():
                    temp_mapping = mapping_old if "end" in item.keys() else mapping_new
                    if index is not None and item[temp_mapping['action']][0] != index:
                        continue
                    if val_topology:
                        
                        if len(item[temp_mapping["topology_end"]]) <= low_limit or len(item[temp_mapping["topology_end"]])>max_limit:
                            continue
                        if len(item[temp_mapping["topology_end"]]) != len(item[temp_mapping['topology_start']])+2:
                            continue
                        temp_list = [None]*20
                        temp_list[:len(item[temp_mapping["topology_end"]])] = item[temp_mapping["topology_end"]][:]
                        topology.append(temp_list)
                    start = np.array(item[temp_mapping['start_config']])
                    gt = np.array(item[temp_mapping['end_pos']])
                    start = start[:config_length]
                    action = convert_action_from_index_to_one_hot_vector(item[temp_mapping['action']],num_of_links)
                    
                    gt[:,:2] -= start[:2]
                    start = fix_yaw_problem(start, config_length)
                    sample = np.concatenate((start, action))
                    #adding init position to sample
                    if return_with_init_position:
                        init_position = np.array(item[temp_mapping['start_pos']])
                        init_position[:,:2] -= start[:2]
                        init_position = np.reshape(init_position, (-1))
                        sample = np.concatenate((sample, init_position))
                    sample[:2] = 0.0
                    gt = np.array(gt)
                    gt = np.reshape(gt, (-1))
                    sample = np.float32(sample)
                    dataset_list.append(list(sample))
                    gt = np.float32(gt)
                    gt_list.append(list(gt))

                    if return_qpos:
                        qpos = np.array(item[temp_mapping['end_config']])
                        qpos = qpos[:config_length]
                        qpos[:2] = qpos[:2] - start[:2]
                        qpos = fix_yaw_problem(qpos, config_length)
                        qpos = np.float32(qpos)
                        qpos_list.append(list(qpos))


                fp.close()

            elif isinstance(dataset_dict, list):
                for item in dataset_dict:
                    temp_mapping = mapping_old if "end" in item.keys() else mapping_new
                    if index is not None and item[temp_mapping['action']][0] != index:
                        continue
                    if val_topology:
                        if len(item[temp_mapping["topology_end"]]) <= low_limit or len(item[temp_mapping["topology_end"]])>max_limit:
                            continue
                        if len(item[temp_mapping["topology_end"]]) != len(item[temp_mapping['topology_start']])+2:
                            continue
                        temp_list = [None]*20
                        temp_list[:len(item[temp_mapping["topology_end"]])] = item[temp_mapping["topology_end"]][:]
                        topology.append(temp_list)
                    start = np.array(item[temp_mapping['start_config']])
                    gt = np.array(item[temp_mapping['end_pos']])
                    start = start[:config_length]
                    action = convert_action_from_index_to_one_hot_vector(item[temp_mapping['action']], num_of_links)
                    
                    gt[:,:2] -= start[:2]
                    start = fix_yaw_problem(start, config_length)
                    sample = np.concatenate((start, action))
                    #adding init position to sample
                    if return_with_init_position:
                        init_position = np.array(item[temp_mapping['start_pos']])
                        init_position[:,:2] -= start[:2]
                        init_position = np.reshape(init_position, (-1))
                        sample = np.concatenate((sample, init_position))
                    sample[:2] = 0.0
                    gt = np.array(gt)
                    gt = np.reshape(gt, (-1))
                    sample = np.float32(sample)
                    dataset_list.append(list(sample))
                    gt = np.float32(gt)
                    gt_list.append(list(gt))

                    if return_qpos:
                        qpos = np.array(item[temp_mapping['end_config']])
                        qpos = qpos[:config_length]
                        qpos[:2] = qpos[:2] - start[:2]
                        qpos = fix_yaw_problem(qpos, config_length)
                        qpos = np.float32(qpos)
                        qpos_list.append(list(qpos))
                fp.close()
            
    if return_qpos:
        dataset = [(dataset_list[i], gt_list[i], qpos_list[i]) for i in range(len(dataset_list))]
    else:
        dataset = [(dataset_list[i], gt_list[i]) for i in range(len(dataset_list))]
    return dataset, topology

def transformer_preprocessing(dataset, num_of_links=21):
    for sample in dataset:
        temp=1
