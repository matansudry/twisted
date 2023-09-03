#packges
import numpy as np
import argparse
import argparse
import os
import pickle
import copy

import sys
sys.path.append(".")

#files

from dm_control import mujoco
from utils.general_utils import *
from utils.topology.state_2_topology import state2topology
from system_flow.system_utils.config_methods import load_config
from utils.enums import RopeAssetsPaths
from data_collection.trejctory_collection import create_dataset_using_trejectory, create_dataset_from_state_using_trejectory

def create_dataset_from_exisiting_datasets():
    args = argparse_create()

    #os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus_to_use)

    import torch
    if not torch.cuda.is_available():
        raise
    torch.backends.cudnn.benchmark = True
    start_seed = args.seed
    physics = mujoco.Physics.from_xml_path(args.env_path)
    physics_reset(physics)
    save_index = 0
    output_location = "datasets/"
    if args.stage == 1:
        for seed in range(start_seed,start_seed+50):
            np.random.seed(seed)
            samples = {}
            name = output_location+"stage_"+str(args.stage)+"_run_"+str(save_index)+"_seed_"+str(seed)+".txt"
            if os.path.isfile(name):
                continue
            index=0
            while(len(samples)<3):
                try:
                    start = get_observation(physics)
                    start_state = physics.get_state()
                    action = get_random_action(min_index=0, max_index=21, max_height=0.07,\
                        min_location=-0.5, max_location=0.5)
                    physics = execute_action_in_curve_with_mujoco_controller(physics, action, get_video=args.get_video, show_image=args.show_image, save_video=False, return_render=False, sample_rate=20)
                    end = get_observation(physics)
                    sample = {
                        "start": start.tolist(),
                        "end": end.tolist(),
                        "action": action.tolist(),
                        "start_data": start_state,
                        "end_data": physics.get_state()
                    }
                    samples[index] = sample
                    index+=1
                    print(index)

                    #set x,y to 0
                    current_state = copy.copy(physics.get_state())
                    current_state[:2] = 0
                    #current_state[47:] = 0
                    physics.reset()
                    with physics.reset_context():
                        physics.set_state(current_state)
                
                except:
                    physics.reset()
                    physics_reset(physics)
                

            if len(samples) > 0:
                with open(name, "wb") as fp:
                    pickle.dump(samples, fp)
                    save_index+=1

    else:
        for seed in range(start_seed,start_seed+50):
            files = os.listdir("datasets/stage_1")
            for file in files:
                if "_seed_"+str(seed) in file:
                    with open("datasets/stage_1/"+file, "rb") as fp:   # Unpickling
                        dataset_dict = pickle.load(fp)
                        for item in dataset_dict.values():
                            if check_topology_state(physics, item['end_data'], args.stage):
                                old_state = item['end_data']
                                with physics.reset_context():
                                    physics.set_state(old_state)
                                for new_seed in range(start_seed,start_seed+50):
                                    np.random.seed(new_seed)
                                    samples = {}
                                    name = output_location+"stage_"+str(args.stage)+"_run_"+str(save_index)+"_seed_"+str(new_seed)+".txt"
                                    if os.path.isfile(name):
                                        continue
                                    index=0
                                    while(len(samples)<10):
                                        try:
                                            start = get_observation(physics)
                                            start_state = physics.get_state()
                                            action = get_random_action(min_index=0, max_index=21, max_height=0.07,\
                                                min_location=-0.5, max_location=0.5)
                                            physics = execute_action_in_curve_with_mujoco_controller(physics, action, args, save_video=args.get_video, return_render=False, sample_rate=20)
                                            end = get_observation(physics)
                                            sample = {
                                                "start": start.tolist(),
                                                "end": end.tolist(),
                                                "action": action.tolist(),
                                                "start_data": start_state,
                                                "end_data": physics.get_state()
                                            }
                                            samples[index] = sample
                                            index+=1

                                            #set x,y to 0
                                            current_state = copy.copy(physics.get_state())
                                            current_state[:2] = 0
                                            #current_state[47:] = 0
                                            physics.reset()
                                            with physics.reset_context():
                                                physics.set_state(current_state)
                                        
                                        except:
                                            with physics.reset_context():
                                                physics.set_state(old_state)
                                        

                                    if len(samples) > 0:
                                        with open(name, "wb") as fp:
                                            pickle.dump(samples, fp)
                                            save_index+=1

def create_dataset_using_exploration():
    args = argparse_create()
    cfg = load_config(path=args.cfg)
    #os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus_to_use)

    args.env_path = RopeAssetsPaths[cfg["STATE2STATE_PARMS"]["num_of_links"]]

    print("num of links is ", cfg["STATE2STATE_PARMS"]["num_of_links"])
    print("rope is ", args.env_path)

    import torch
    if not torch.cuda.is_available():
        raise
    torch.backends.cudnn.benchmark = True
    start_seed = args.seed
    physics = mujoco.Physics.from_xml_path(args.env_path)
    #print(physics.named.data.geom_xpos)
    physics_reset(physics)
    num_of_links = get_number_of_links(physics=physics)
    while(1):
        cnt = 0
        for seed in range(start_seed,start_seed+50):
            if cnt >= 21:
                cnt = 0
                physics.reset()
                physics_reset(physics)
            set_seed(seed)
            #try:
            start = get_position_from_physics(physics)
            start_state = physics.get_state()
            action = get_random_action(min_index=0, max_index=num_of_links-1, max_height=0.07,\
                min_location=-0.35, max_location=0.35)
            physics = execute_action_in_curve_with_mujoco_controller(physics, action, num_of_links=num_of_links, get_video=args.get_video,\
                 show_image=args.get_image, save_video=False, return_render=False, sample_rate=20, output_path = "outputs/videos/15_links/")
            end = get_position_from_physics(physics)
            topology_end = state2topology(torch.tensor(end))
            topology_end = topology_end #[0].points
            topology_start = state2topology(torch.tensor(start))
            topology_start = topology_start #[0].points

            #we have 1-5 crosses
            if len(topology_end) > 2 and len(topology_end) < 13:
                sample = {
                    "start": start.tolist(),
                    "end": end.tolist(),
                    "action": action.tolist(),
                    "start_data": start_state,
                    "end_data": physics.get_state(),
                    "topology_start": topology_start,
                    "topology_end": topology_end
                }
                name = get_file_name(int((len(topology_end)-2)/2),dataset_path="datasets/15_links_new")
                with open(name, "wb") as fp:
                    pickle.dump([sample], fp)
                cnt+=1

if __name__ == '__main__':
    create_dataset_from_state_using_trejectory()