import sys
sys.path.append(".")

#packges
import os
import pickle
import copy
from dm_control import mujoco
import argparse
import tqdm

#files
from utils.general_utils import get_number_of_links, set_seed, get_position_from_physics,\
    execute_action_in_curve_with_mujoco_controller, set_physics_state, get_random_action
from utils.topology.state_2_topology import state2topology
from system_flow.system_utils.config_methods import load_config
from data_collection.data_collection_utils import reset_physics_to_init_state, save_trejctory,\
    convert_topology_to_number_of_crosses

def trejctory_collection_automation(args:dict):
    """
    collect data with resets
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
        reset_physics_to_init_state(physics)
        current_cross_state = 0
        trejctory = []
        for _ in range(2):
            for seed in range(start_seed,start_seed+50):
                set_seed(seed+1)
                start_config = copy.deepcopy(physics.get_state())
                start_pos = get_position_from_physics(physics)
                action = get_random_action(
                    min_index=0,
                    max_index=num_of_links-1,
                    max_height=0.07,
                    min_location=-0.5,
                    max_location=0.5
                    )
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
                temp_number_of_crosses = convert_topology_to_number_of_crosses(topology_end)
                
                #if the action was not good
                if temp_number_of_crosses != current_cross_state+1:
                    set_physics_state(physics, start_config)
                    continue
                
                topology_start = state2topology(torch.tensor(start_pos))
                sample = {
                    "start_config": start_config,
                    "end_config": copy.deepcopy(physics.get_state()),
                    "action": action.tolist(),
                    "start_pos": start_pos,
                    "end_pos": end_pos,
                    "topology_start": topology_start,
                    "topology_end": topology_end
                }
                trejctory.append(copy.deepcopy(sample))
                current_cross_state +=1
                #we reached goal
                if (args.trejctory_length == current_cross_state):
                    save_trejctory(
                        trejctory=trejctory,
                        main_folder=args.main_folder,
                        num_of_crosses=args.trejctory_length
                        )
                    reset_physics_to_init_state(physics)
                    current_cross_state = 0
                    trejctory = []

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
    parser.add_argument('--main_folder', type=str, default="datasets/21_links/")

    args = parser.parse_args()


    trejctory_collection_automation(args=args)