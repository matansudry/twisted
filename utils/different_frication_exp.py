import sys
sys.path.append(".")

import yaml
import os
import numpy as np

from utils.general_utils import update_parms, main_argparse
from data_collection.data_collection_utils import *
from utils.general_utils import *
import multiprocessing
import time

def check_different_frication(start_config, new_action, output_path, env_paths:list):
    topology_states = {}
    for env_path in env_paths:
        physics = mujoco.Physics.from_xml_path(env_path)
        #check continues
        set_physics_state(physics, start_config)
        physics = execute_action_in_curve_with_mujoco_controller(physics, new_action, num_of_links=21,\
                                                        get_video=False, show_image=False,\
                                                        save_video=False, return_render=False, sample_rate=20,\
                                                        output_path = "outputs/videos/15_links/",
                                                        env_path=env_path)
        end_pos = get_position_from_physics(physics)
        topology_states[env_path] = state2topology(torch.tensor(end_pos))

    output = {}
    for env_path in env_paths:
        output[env_path] = {}
        for env_path_second in env_paths:
            if env_path == env_path_second:
                continue
            output[env_path][env_path_second] = comperae_two_high_level_states(topology_states[env_path], topology_states[env_path_second])
    
    #save
    save_pickle(output_path, output)

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
            max_limit=cfg["main"]["train_max_cross"])#, num_of_samples=100)


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

        path = "exp/different_frication/"

        full_path = os.path.join(path,str(index)+".pickle")

        env_paths = [
            "assets/rope_v3_21_links.xml",
            "assets/rope_v3_21_links_friction_95.xml",
            "assets/rope_v3_21_links_friction_105.xml",
        ]

        #check action
        p = multiprocessing.Process(target=check_different_frication, args=(start_config, new_action, full_path, env_paths))
        p.start()
        #output = check_different_frication(start_config, new_action, full_path, env_paths)
        time.sleep(0.0001)
        
    print(outputs)

if __name__ == '__main__':
    args = main_argparse()
    main(args)
