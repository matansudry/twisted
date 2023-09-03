from re import X
import sys
sys.path.append(".")

import yaml
import os
import numpy as np

from utils.general_utils import update_parms, main_argparse
from utils.enums import RopeAssetsPaths
from data_collection.data_collection_utils import *
from utils.general_utils import *
import multiprocessing
import time

def change_action_to_discrete(action, height_resolution=0.001, height_range=[0.001, 0.07], x_resolution=0.01, x_range=[-0.5, 0.5],\
    y_resolution=0.01, y_range=[-0.5, 0.5]):
    #fix height
    height = action[1]
    height_diff = height%height_resolution
    height = height-height_diff + height_resolution
    height = min(height, height_range[1])
    height = max(height, height_range[0])
    assert height >= height_range[0] and height <= height_range[1], f'height is wrong {height}'

    #fix x
    x = action[2]
    x_diff = x%x_resolution
    x = x-x_diff + x_resolution
    x = min(x, x_range[1])
    x = max(x, x_range[0])
    assert x >= x_range[0] and x <= x_range[1], f'x is wrong {x}'

    #fix y
    y = action[3]
    y_diff = y%y_resolution
    y = y-y_diff + y_resolution
    y = min(y, y_range[1])
    y = max(y, y_range[0])
    assert y >= y_range[0] and y <= y_range[1], f'y is wrong {y}'

    return np.array([action[0], height,x,y])

def check_discrete_action(start_config, new_action, goal_topology, path):
    env_path = RopeAssetsPaths[21]
    physics = mujoco.Physics.from_xml_path(env_path)
    
    #check continues
    set_physics_state(physics, start_config)
    physics = execute_action_in_curve_with_mujoco_controller(physics, new_action, num_of_links=21,\
                                                    get_video=False, show_image=False,\
                                                    save_video=False, return_render=False, sample_rate=20,\
                                                    output_path = "outputs/videos/15_links/")
    end_pos = get_position_from_physics(physics)
    topology_continues = state2topology(torch.tensor(end_pos))

    #check discrete
    set_physics_state(physics, start_config)
    action = change_action_to_discrete(new_action)
    physics = execute_action_in_curve_with_mujoco_controller(physics, action, num_of_links=21,\
                                                    get_video=False, show_image=False,\
                                                    save_video=False, return_render=False, sample_rate=20,\
                                                    output_path = "outputs/videos/15_links/")
    end_pos = get_position_from_physics(physics)
    topology_discrete = state2topology(torch.tensor(end_pos))

    #calculate matrics
    continues_discrete_equal = comperae_two_high_level_states(topology_continues, topology_discrete)
    continues_goal_equal = comperae_two_high_level_states(topology_continues, goal_topology)
    discrete_goal_equal = comperae_two_high_level_states(topology_discrete, goal_topology)

    #create output
    output = {
        "topology_continues": topology_continues,
        "topology_discrete": topology_discrete,
        "goal_topology": goal_topology,
        "continues_discrete_equal": continues_discrete_equal,
        "continues_goal_equal": continues_goal_equal,
        "discrete_goal_equal": discrete_goal_equal
    }
    
    #save
    save_pickle(path, output)

def main(args) -> None:
    # Read YAML file
    with open(args.cfg, 'r') as stream:
        cfg = yaml.safe_load(stream)

    #os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg['main']['gpus_to_use'])

    os.system('CUDA_LAUNCH_BLOCKING=1')

    import torch
    torch.backends.cudnn.benchmark = True

    torch.multiprocessing.set_sharing_strategy('file_system')
    from state2state_flow.s2s_utils import main_utils
    from state2state_flow.s2s_utils.dataset_utils import preprocessing_dataset_qpos

    cfg = update_parms(args,cfg)

    cfg["main"]["config_length"] = (cfg["main"]["num_of_links"]-1)*2+7
    #config + hot vectore action + x,y,height + position num_of_links*3
    cfg['train']['input_size'] = cfg["main"]["config_length"] + cfg["main"]["num_of_links"] + 3 #+ ((cfg["main"]["num_of_links"] +1)*3)
    cfg['train']['output_size'] = cfg["main"]["config_length"]
    if cfg["main"]["return_with_init_position"]:
        cfg['train']['input_size'] += 3*(cfg["main"]["num_of_links"]+1)
    
    cfg['main']['experiment_name_prefix']=\
        cfg['main']['experiment_name_prefix']+"_seed="+str(cfg['main']['seed'])+"_lr_value="+str(cfg['train']['lr']["lr_value"])+"_"

    with_qpos = cfg['main']["with_qpos"]

    #Load dataset
    train_dataset, train_topology_list = preprocessing_dataset_qpos(cfg, paths=cfg['main']['paths']['train'], val_topology=True,\
            return_qpos=with_qpos,low_limit=cfg["main"]["train_min_cross"], return_with_init_position=cfg["main"]["return_with_init_position"],\
            max_limit=cfg["main"]["train_max_cross"]) #, num_of_samples=100)


    length = 137
    config_length = length - 3 -  cfg["main"]["num_of_links"] #len(train_dataset[0][0]) - 3 -  cfg["main"]["num_of_links"]
    if (cfg['train']['input_size'] != length): #len(train_dataset[0][0])):# and config_length !=cfg["main"]["config_length"]:
        print("cfg['train']['input_size'] = ", cfg['train']['input_size'])
        print("len(train_dataset[0][0]) = ", length) #len(train_dataset[0][0]))
        print("data config_length = ", config_length)
        print("cfg['main']['config_length'] = ", cfg["main"]["config_length"])
        raise


    outputs = []
    args = []
    
    for index, sample in tqdm.tqdm(enumerate(train_dataset)):
        start_config = np.zeros(93)
        raw_start_config = np.array(sample[0][:47])
        start_config[:47] = raw_start_config
        raw_action = np.array(sample[0][47:71])
        action_index = np.argmax(raw_action[:21])
        new_action = np.array([action_index, raw_action[21], raw_action[22], raw_action[23]])
        goal_topology = train_topology_list[index]
        goal_topology = [i for i in goal_topology if i is not None]

        path = "exp/dicrete_results/"+str(index)+".pickle"

        #check action
        p = multiprocessing.Process(target=check_discrete_action, args=(start_config, new_action, goal_topology, path))
        p.start()
        #output = check_discrete_action(start_config, new_action, goal_topology, path)
        time.sleep(0.1)
        
    print(outputs)

if __name__ == '__main__':
    args = main_argparse()
    main(args)
