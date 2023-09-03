import sys
sys.path.append(".")

#packges
import numpy as np
import argparse
import os
import pickle
import copy
import random
import tqdm
from dm_control import mujoco

#files
from utils.general_utils import get_number_of_links, set_physics_state, set_seed, get_position_from_physics,\
    execute_action_in_curve_with_mujoco_controller
from utils.topology.state_2_topology import state2topology
from system_flow.system_utils.config_methods import load_config
from data_collection.data_collection_utils import reset_physics_to_init_state,\
    convert_topology_to_number_of_crosses, save_trejctory

def add_noise_to_action(action:list, num_of_links:int) -> list:
    """
    adding noise to action

    Args:
        action (list): original action
        num_of_links (int): how much links there is in rope

    Returns:
        list: new action
    """
    action[0] += random.randint(-1,1)
    if action[0] < 0:
        action[0]+=1
    if action[0] > num_of_links-1:
        action[0] -= 1

    height_random_number = (random.random() - 0.5)*0.1
    action[1] += height_random_number
    if action[1] < 0:
        action[1] = 0.00001
    if action[1] > 0.07:
        action[1] = 0.07 
    x_random_number = (random.random() - 0.5)*0.1
    action[2] += x_random_number
    y_random_number = (random.random() - 0.5)*0.1
    action[3] += y_random_number

    return action

def action_noise_automation(args:dict):
    """
    Load sample and add noise on it
    """
    cfg = load_config(path=args.cfg)
    cfg["STATE2STATE_PARMS"]["num_of_links"] = 21

    print("num of links is ", cfg["STATE2STATE_PARMS"]["num_of_links"])
    print("rope is ", args.env_path)

    import torch
    if not torch.cuda.is_available():
        raise
    torch.backends.cudnn.benchmark = True
    start_seed = args.seed
    physics = mujoco.Physics.from_xml_path(args.env_path)
    num_of_links = get_number_of_links(physics=physics)
    
    while(1):
        for load_data_path in args.load_data_paths:
            files = os.listdir(load_data_path)
            random.shuffle(files)
            for file in tqdm.tqdm(files):
                with open(load_data_path+file, "rb") as fp:
                    dataset_dict = pickle.load(fp)
                fp.close()
                initial_state = dataset_dict[0]["start_config"]
                reset_physics_to_init_state(physics)
                set_physics_state(physics, initial_state)
                trejctory = []
                state_found = False
                for _ in range(100):
                    if state_found:
                        break
                    for seed in range(start_seed,start_seed+50):
                        if state_found:
                            break
                        set_seed(seed+args.add_number_to_seed)
                        start_pos = get_position_from_physics(physics)
                        action = copy.deepcopy(dataset_dict[0]['action'])
                        action = add_noise_to_action(action, num_of_links)
                        action = np.array(action)
                        physics = execute_action_in_curve_with_mujoco_controller(
                            physics=physics,
                            action=action,
                            num_of_links=num_of_links,
                            get_video=args.get_video,
                            show_image=args.get_image,
                            save_video=False,
                            return_render=False,
                            sample_rate=20,
                            output_path=args.video_output_path
                            )
                        end_pos = get_position_from_physics(physics)
                        topology_end = state2topology(torch.tensor(end_pos))
                        end_number_of_crosses = convert_topology_to_number_of_crosses(topology_end)
                        
                        #if the action was not good
                        if end_number_of_crosses != args.trejctory_length:
                            set_physics_state(physics, initial_state)
                            continue
                        
                        topology_start = state2topology(torch.tensor(start_pos))
                        sample = {
                            "start_config": initial_state,
                            "end_config": copy.deepcopy(physics.get_state()),
                            "action": action.tolist(),
                            "start_pos": start_pos,
                            "end_pos": end_pos,
                            "topology_start": topology_start,
                            "topology_end": topology_end
                        }
                        trejctory.append(copy.deepcopy(sample))
                        #we reached goal
                        if (args.trejctory_length == end_number_of_crosses):
                            save_trejctory(
                                trejctory=trejctory,
                                main_folder=args.output_path,
                                num_of_crosses=args.trejctory_length,
                                prefix=args.output_prefix
                                )
                            reset_physics_to_init_state(physics)
                            trejctory = []
                            state_found = True
                            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_path', type=str, default="assets/rope_v3_21_links.xml")
    parser.add_argument('--cfg', type=str, default='system_flow/config/system_config.yml')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_path', type=str, default="datasets/21_links/")
    parser.add_argument('--load_data_paths', nargs="+")
    parser.add_argument('--add_number_to_seed', type=int, default=101)
    parser.add_argument('--trejctory_length', type=int, default=3)
    parser.add_argument('--output_prefix', type=str, default="action_noise_batch_4_")
    parser.add_argument('--video_output_path', type=str, default="outputs/videos/15_links/")

    args = parser.parse_args()

    action_noise_automation()

