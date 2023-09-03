import sys
sys.path.append(".")

import numpy as np
import pytorch3d.transforms as pyt
import copy
import torch
from dm_control import mujoco
import tqdm

from utils.general_utils import set_physics_state, get_position_from_physics,\
    convert_action_from_index_to_one_hot_vector
from utils.topology.state_2_topology import state2topology

def rotate_state(qpos, angle, principal_axe="yaw"):
    axe_map = {
        "yaw": 0,
        "pitch": 1,
        "roll": 2
    }
    new_qpos = copy.copy(qpos)
    convention = "ZXY"
    quaternion = torch.tensor(copy.copy(new_qpos[3:7]))
    rotation = torch.ones([3,3])
    rotation[:,:] *= pyt.quaternion_to_matrix(quaternion)
    euler = pyt.matrix_to_euler_angles(rotation, convention)
    euler[axe_map[principal_axe]] += angle
    new_rotation = pyt.euler_angles_to_matrix(euler, convention)
    new_quaternion = pyt.matrix_to_quaternion(new_rotation)
    new_quaternion = new_quaternion.tolist()
    new_qpos[3:7] = new_quaternion[:]
    return new_qpos

def rotate_action(x, y, angle):
    new_x = np.cos(angle)*x - np.sin(angle)*y
    new_y = np.sin(angle)*x + np.cos(angle)*y
    return new_x, new_y

def add_augmantations(dataset, topology):
    env_path = "assets/rope_v3_21_links.xml"
    physics = mujoco.Physics.from_xml_path(env_path)
    final_dataset = copy.copy(dataset)
    final_topology = copy.copy(topology)   
    for angle in [np.pi/2, np.pi, np.pi*3/2]:
        new_dataset = []
        for index in tqdm.tqdm(range(len(dataset))):
            #need to update
            temp_sample = dataset[index][0]
            #need to update
            temp_gt = dataset[index][1]
            #need to update
            temp_end_qpos = dataset[index][2]
            old_start_qpos = np.zeros(93)
            old_start_qpos[:47] = temp_sample[:47]
            #get start qpos
            new_start_qpos = rotate_state(old_start_qpos, angle=angle)

            #get new start pos
            set_physics_state(physics, new_start_qpos)
            new_start_pos = get_position_from_physics(physics)
            new_start_pos = np.reshape(new_start_pos, (-1))

            #get start action
            old_action = temp_sample[47:71]
            temp_new_action = [0, old_action[-3], old_action[-2], old_action[-1]]
            for temp_index in range(21):
                if old_action[temp_index] == 1:
                    temp_new_action[0] = temp_index
                    break
            temp_new_action[2], temp_new_action[3] = rotate_action(x=temp_new_action[2], y=temp_new_action[3], angle=angle)
            new_action = convert_action_from_index_to_one_hot_vector(temp_new_action,num_of_links=21)

            sample = np.concatenate((new_start_qpos[:47], new_action))
            sample = np.concatenate((sample, new_start_pos))

            new_end_qpos = rotate_state(temp_end_qpos, angle=angle)
            temp_new_end_qpos = np.zeros(93)
            temp_new_end_qpos[:47] = new_end_qpos[:47]
            #set end qpos to get new gt (pos)
            set_physics_state(physics, np.array(temp_new_end_qpos))
            new_gt = get_position_from_physics(physics)
            new_gt = np.reshape(new_gt, (-1))

            new_dataset.append((list(sample), list(new_gt), new_end_qpos))
        final_dataset.extend(copy.copy(new_dataset))
        final_topology.extend(copy.copy(topology))
    return final_dataset, final_topology

def reverse_state(state, num_of_links=21):
    temp_state = copy.deepcopy(state)
    left = copy.deepcopy(temp_state[7+num_of_links-1:7+num_of_links-1+num_of_links-1])
    right = copy.deepcopy(temp_state[7:num_of_links+7-1])
    for index in range(len(left)):
        if index%2 == 1:
            continue
        left[index] *= -1
        right[index] *= -1  
    temp_state[7:27] = copy.copy(left)[:]
    temp_state[27:47] = copy.copy(right)[:]
    return temp_state

def reverse_action(index, num_of_links=21):
    new_index = num_of_links-1-index
    return new_index

def add_augmantations_reverse(dataset, topology):
    env_path = "assets/rope_v3_21_links.xml"
    physics = mujoco.Physics.from_xml_path(env_path)
    final_dataset = copy.copy(dataset)
    final_topology = copy.copy(topology)   
    new_dataset = []#copy.copy(dataset)
    new_topology = []#copy.copy(topology)
    for index in tqdm.tqdm(range(len(dataset))):
        #need to update
        temp_sample = dataset[index][0]
        #need to update
        temp_gt = dataset[index][1]
        #need to update
        temp_end_qpos = dataset[index][2]
        old_start_qpos = np.zeros(93)
        old_start_qpos[:47] = temp_sample[:47]
        #get start qpos
        set_physics_state(physics, old_start_qpos)

        new_start_qpos = rotate_state(old_start_qpos, angle=np.pi, principal_axe="yaw")
        new_start_qpos = reverse_state(new_start_qpos)
        set_physics_state(physics, new_start_qpos)

        new_start_pos = get_position_from_physics(physics)
        new_start_pos = np.reshape(new_start_pos, (-1))

        #get start action
        old_action = temp_sample[47:71]
        temp_new_action = [0, old_action[-3], old_action[-2], old_action[-1]]
        for temp_index in range(21):
            if old_action[temp_index] == 1:
                temp_new_action[0] = temp_index
                break
        temp_new_action[0] = reverse_action(temp_new_action[0])
        new_action = convert_action_from_index_to_one_hot_vector(temp_new_action,num_of_links=21)

        sample = np.concatenate((new_start_qpos[:47], new_action))
        sample = np.concatenate((sample, new_start_pos))

        #new_end_qpos = rotate_state(temp_end_qpos, angle=np.pi)

        new_end_qpos = rotate_state(temp_end_qpos, angle=np.pi, principal_axe="yaw")
        new_end_qpos = reverse_state(new_end_qpos)

        temp_new_end_qpos = np.zeros(93)
        temp_new_end_qpos[:47] = new_end_qpos[:47]
        #set end qpos to get new gt (pos)
        set_physics_state(physics, np.array(temp_new_end_qpos))
        new_gt = get_position_from_physics(physics)
        new_gt = np.reshape(new_gt, (-1))

        new_topology_gt = state2topology(torch.tensor(new_gt))
        new_topology.append(new_topology_gt)

        new_dataset.append((list(sample), list(new_gt), new_end_qpos))
    final_dataset.extend(copy.copy(new_dataset))
    final_topology.extend(copy.copy(new_topology))
    return final_dataset, final_topology