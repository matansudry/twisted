#packages
import numpy as np
import torch
import math
from shapely.geometry import LineString
import copy
import tqdm
import pytorch3d.transforms as pyt
import cv2
import argparse
import pandas as pd
import os
import seaborn as sn
import matplotlib.pyplot as plt
import pickle
from dm_control import mujoco
from multiprocessing import Process
from time import sleep
from easydict import EasyDict as edict


#files
from utils.topology.state_2_topology import state2topology
from state2state_flow.s2s_utils.metrics import conf_mat_score_funcation_topology #score_function_mse, score_funcation_topology,
#from utils.enums import RopeAssetsPaths

def main_argparse():
    parser = argparse.ArgumentParser()

    # Data Parameters
    parser.add_argument('--cfg', type=str, default="state2state_flow/config/config.yaml")

    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--lr_value', type=float, default=None)
    parser.add_argument('--seed', type=float, default=0)

    args = parser.parse_args()
    return args

def argparse_create():
    parser = argparse.ArgumentParser()

    # Data Parameters
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpus_to_use', type=int, default=0)
    parser.add_argument('--env_path', type=str, default=None) #'assets/rope_v4.xml')
    parser.add_argument('--cfg', type=str, default='system_flow/config/system_config.yml')
    parser.add_argument('--load_state2state', type=bool, default=True)
    parser.add_argument('--load_state2action', type=bool, default=True)
    parser.add_argument('--reverse_high_level_plan', type=bool, default=False)
    parser.add_argument('--step_size', type=int, default=0.00001)
    parser.add_argument('--location_k', type=int, default=1)
    parser.add_argument('--velocity_k', type=int, default=1)
    parser.add_argument('--location_z', type=int, default=100000)
    parser.add_argument('--location_x_y', type=int, default=7)
    parser.add_argument('--max_force_x_y', type=float, default=0.1)
    parser.add_argument('--max_force_z', type=int, default=5)
    parser.add_argument('--get_image', type=bool, default=False)
    parser.add_argument('--get_video', type=bool, default=False)
    parser.add_argument('--treshold', type=float, default=0.0010)
    parser.add_argument('--stage', type=bool, default=1)
    parser.add_argument('--problme_index', type=bool, default=0)
    parser.add_argument('--run_time_limit', type=bool, default=1800)
    parser.add_argument('--problems_file', type=str, default="metrics/all_3_crosses_states.txt")


    args = parser.parse_args()

    return args

def calc_logprob(mu_v, logstd_v, actions_v):
    p1 = - ((mu_v - actions_v) ** 2) / (2*torch.exp(logstd_v).clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2 * math.pi * torch.exp(logstd_v)))
    return p1 + p2

def calc_adv_ref(trajectory, net_crt, states_v, config, device="cpu"):
    """
    By trajectory calculate advantage and 1-step ref value
    :param trajectory: trajectory list
    :param net_crt: critic network
    :param states_v: states tensor
    :return: tuple with advantage numpy array and reference values
    """
    values_v = net_crt(states_v.float())
    values = values_v.squeeze().data.cpu().numpy()
    # generalized advantage estimator: smoothed version of the advantage
    last_gae = 0.0
    result_adv = []
    result_ref = []

    if (len(trajectory) == 1):
        val = 0
        next_val = values.item()
        (exp,) = trajectory[0]
        if exp.done:
            delta = exp.reward - val
            last_gae = delta
        else:
            delta = exp.reward + config["NETWORK_PARMS"]["GAMMA"] * next_val - val
            last_gae = delta + config["NETWORK_PARMS"]["GAMMA"] * config["NETWORK_PARMS"]["GAE_LAMBDA"] * last_gae
        result_adv.append(last_gae)
        result_ref.append(last_gae + val)
        adv_v = torch.FloatTensor(list(reversed(result_adv))).to(device)
        ref_v = torch.FloatTensor(list(reversed(result_ref))).to(device)
    else:
        for val, next_val, (exp,) in zip(reversed(values[:-1]), reversed(values[1:]),
                                        reversed(trajectory[:-1])):
            if exp.done:
                delta = exp.reward - val
                last_gae = delta
            else:
                delta = exp.reward + config["GAMMA"] * next_val - val
                last_gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * last_gae
            result_adv.append(last_gae)
            result_ref.append(last_gae + val)

        adv_v = torch.FloatTensor(list(reversed(result_adv))).to(device)
        ref_v = torch.FloatTensor(list(reversed(result_ref))).to(device)
    return adv_v, ref_v

def intersection (first_line_start_point, first_line_end_point, second_line_start_point, second_line_end_point):
    """
    this method getting 4 points, 2 from each line and checking if there is intersection between them
    """
    start_line =  LineString([first_line_start_point, first_line_end_point])
    end_line = LineString([second_line_start_point, second_line_end_point])
    intersection_lines = start_line.intersection(end_line)
    return intersection_lines.bounds

def create_spline_from_points(start_point, max_height, end_point, step_size = 0.01):
    """
    This method will get a 3 points and step size and return location and velocity in each time stamp
    """
    # start and end point are in z=0

    second_point = copy.copy(start_point)
    second_point[2] = max_height
    third_point = copy.copy(end_point)
    third_point[2] = max_height

    points_1 = []
    points_2 = []
    points_3 = []
    num_of_points = max(int((max_height - start_point[2])/step_size), int(0.01/step_size))
    z_points = np.linspace(start_point[2], max_height, num_of_points)
    x_y_max_points = int(max(abs(end_point[0] - start_point[0]), abs(end_point[1] - start_point[1])) / step_size)

    #from start to second point
    points_1.append(list(start_point))
    for item in z_points:
        points_1.append([start_point[0], start_point[1], item])
    
    #from second point to third point
    x_points = np.linspace(start_point[0], end_point[0], x_y_max_points)
    y_points = np.linspace(start_point[1], end_point[1], x_y_max_points)
    for index in range(x_y_max_points):
        points_2.append([x_points[index], y_points[index], max_height])

    #from third to end
    for item in reversed(z_points):
        points_3.append([end_point[0], end_point[1], item])
    
    points = {}
    points["location"] = [points_1, points_2, points_3]
    return points

def get_random_action(min_index, max_index, max_height, min_location, max_location):
    index = np.random.randint(low=min_index, high=max_index)
    #max_height_value = np.random.uniform(low=min_height, high=max_height)
    end_location = np.random.uniform(low=min_location, high=max_location, size=2)
    return np.array([index, max_height, end_location[0], end_location[1]])

def location_and_velocity_reached(physics, index, position=True, velocity=True, velo_tol=5, loc_tol=0.0005, max_velo = 0.6):
    if position:
        goal_location = copy.copy(physics._data.mocap_pos[0])
        current_location = copy.copy(physics._data.xpos[index])

        if pow(goal_location[0]-current_location[0],2) > loc_tol or\
            pow(goal_location[1]-current_location[1],2) > loc_tol or\
            pow(goal_location[2]-current_location[2],2) > loc_tol:
            return False
    if velocity:
        if sum(pow(physics.data.qvel[:],2)) > velo_tol:
            return False

        if max(physics.data.qvel[:]) > max_velo or abs(min(physics.data.qvel[:])) > max_velo:
            return False
    
    return True

def update_mocap_location(physics, action_index):
    physics.model.eq_obj2id[:] = [action_index]
    physics._data.mocap_pos[:] = copy.copy(physics.data.xpos[action_index])
    physics._data.mocap_quat[:] = copy.copy(physics.data.xquat[action_index])
    physics.model.eq_data[0][:6] = [0,0,0,0,0,0] 

def delete_mocap(physics):
    physics.model.eq_active[:] = 0

def enable_mocap(physics):
    physics.model.eq_active[:] = 1

def convert_name_to_index(name, num_of_links=None):
    if num_of_links == 21:
        names = {
            "G0": 21,"G1": 20,"G2": 19,"G3": 18,"G4": 17,
            "G5": 16,"G6": 15,"G7": 14,"G8": 13,"G9": 12,
            "G10": 1,"G11": 2,"G12": 3,"G13": 4,"G14": 5,
            "G15": 6,"G16": 7,"G17": 8,"G18": 9,"G19": 10,"G20": 11
        }
    elif num_of_links == 15:
        names = {
            "G0": 15,"G1": 14,"G2": 13,"G3": 12,"G4": 11,
            "G5": 10,"G6": 9,"G7": 1,"G8": 2,"G9": 3,
            "G10": 4, "G11": 5, "G12": 6, "G13": 7,
            "G14": 8
        }
    elif num_of_links == 11:
        names = {
            "G0": 11,"G1": 10,"G2": 9,"G3": 8,"G4": 7,
            "G5": 1,"G6": 2,"G7": 3,"G8": 4,"G9": 5,
            "G10": 6
        }
    elif num_of_links == 7:
        names = {
            "G0": 7,"G1": 6,"G2": 5,"G3": 1,"G4": 2,
            "G5": 3,"G6": 4
        }
    else:
        raise
    return names[name]

def execute_action_in_curve_with_mujoco_controller(physics, action, num_of_links, get_video=False,\
     show_image=False, save_video=False,\
     return_render=True, sample_rate=20, video=None, output_path="outputs/videos/", env_path="", return_video=False):

    #reset physics
    current_state = get_current_primitive_state(physics)
    num_of_links = get_number_of_links(physics=physics, qpos=current_state)
    
    if env_path == "":
        env_path = RopeAssetsPaths[num_of_links]
    
    physics = mujoco.Physics.from_xml_path(env_path)
    set_physics_state(physics, current_state)

    enable_mocap(physics)
    video = []

    joint = "G"+str(int(action[0]))
    action_index = convert_name_to_index(joint,num_of_links=num_of_links)

    #update the locatino of the mocap according the link that will be move
    update_mocap_location(physics, action_index)
    tries = 0
    while not location_and_velocity_reached(physics, action_index, position=False, velocity=True, velo_tol=2,\
         loc_tol=0.0001, max_velo = 0.1) and tries < 1000:
        physics.step()
        tries += 1
        if get_video:
            pixels = physics.render()
            video.append(pixels)
        if show_image and index%sample_rate==0:
            cv2.imshow("image", pixels)
            cv2.waitKey(2) 

    start_point = physics.data.xpos[action_index]
    end_point= [action[2] + start_point[0],action[3] + start_point[1], 0]
    max_height = action[1]
    if video is None:
        video = []

    splines = create_spline_from_points(start_point=start_point, max_height=max_height, end_point=end_point,\
         step_size=0.00001)

    number_of_paths_in_curve = len(splines["location"])

    for path_in_spline in range(number_of_paths_in_curve):
        position = splines["location"][path_in_spline]
        for index in range(len(position)-1):
            physics._data.mocap_pos[:] = position[index]
            physics._data.mocap_quat[:] = copy.copy(physics.data.xquat[action_index])
            tries = 0
            while not location_and_velocity_reached(physics, action_index, position=True, velocity=False,\
                 loc_tol=0.0001) and tries < 1000:
                tries += 1
                physics.step()
                if get_video:# and index%sample_rate==0:
                    pixels = physics.render()
                    video.append(pixels)
                if show_image and index%sample_rate==0:
                    cv2.imshow("image", pixels)
                    cv2.waitKey(2)
        
        #last action in the curve
        physics._data.mocap_pos[0] = position[-1]
        tries = 0
        while not location_and_velocity_reached(physics, action_index, position=True, velocity=True) and tries < 1000:
            tries += 1
            physics.step()
            if get_video:
                pixels = physics.render()
                video.append(pixels)
            if show_image and index%sample_rate==0:
                cv2.imshow("image", pixels)
                cv2.waitKey(2)
    physics._data.mocap_quat[:] = copy.copy(physics.data.xquat[action_index])
    tries = 0
    while not location_and_velocity_reached(physics, action_index, position=True, velocity=True,\
         velo_tol=2, loc_tol=0.0001, max_velo = 0.1)  and tries < 1000:
        physics.step()
        tries +=1
        if get_video:
            pixels = physics.render()
            video.append(pixels)
        if show_image and index%sample_rate==0:
            cv2.imshow("image", pixels)
            cv2.waitKey(2)  

    delete_mocap(physics)
    physics.step()
    physics._data.mocap_quat[:] = copy.copy(physics.data.xquat[action_index])
    tries = 0
    while not location_and_velocity_reached(physics, action_index, position=False, velocity=True,\
         velo_tol=1.5, loc_tol=0.0001, max_velo = 0.05) and tries < 1000:
        for _ in range(10):
            physics.step()
            if get_video:
                pixels = physics.render()
                video.append(pixels)
            if show_image and index%sample_rate==0:
                cv2.imshow("image", pixels)
                cv2.waitKey(2) 
        tries += 1

    if save_video:
        video_index = 0
        free = True
        while free:
            if os.path.isfile(output_path+str(video_index)+"_project.avi"):
                video_index += 1
            else:
                free=False

        path = output_path+str(video_index)+"_project.avi"
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"DIVX"), 60, (640,480))
        for i in range(len(video)):
            if i%sample_rate ==0:
                out.write(video[i])
        out.write(video[-1])
        out.release()
    
    if return_video:
        return physics, video
    else:
        return physics

def load_pickle(path):
    with open(path, "rb") as fp:
        file = pickle.load(fp)
    fp.close()
    return file

def comperae_two_high_level_states(state_1, state_2):
    if not isinstance(state_1, list):
        state_1 = state_1.points
    if not isinstance(state_2, list):
        state_2 = state_2.points
    if len(state_1) != len(state_2):
        return False
    for index in range(len(state_2)):
        if state_1[index].over != state_2[index].over or\
        state_1[index].sign != state_2[index].sign or\
        state_1[index].under != state_2[index].under:
            return False
    return True

def physics_reset(physics):
    physics.reset()
    num_of_links = get_number_of_links(physics)
    config_length = num_of_links*2 + 7 
    current_state = copy.copy(physics.get_state())
    for index in range(7,config_length):
        if index%2 == 1:
            continue
        if index > 7 + num_of_links:
            current_state[index] -= np.pi / ((config_length-7)/4)
        else:
            current_state[index] += np.pi / ((config_length-7)/4)
    set_physics_state(physics, current_state)        

def update_parms(args,cfg):
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.dropout is not None:
        cfg["train"]["dropout"] = args.dropout
    if args.lr_value is not None:
        cfg["train"]["lr"]["lr_value"] = args.lr_value
    cfg["main"]["seed"] = int(args.seed)
    return cfg

def check_topology_state(physics, state, stage):
    current_state = copy.copy(physics.get_state())
    set_physics_state(physics, state)
    position = copy.copy(physics._data.geom_xpos[:])
    topology = state2topology(torch.tensor(position))
    topology = topology[0].points
    set_physics_state(physics, current_state)
    if len(topology) == stage*2:
        return True
    return False

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def calculate_number_of_crosses_from_topology_state(topology_state):
    return int((len(topology_state.points)-2)/2)

def get_file_name(number_of_crosses, dataset_path="datasets", prefix=""):
    path = dataset_path +"/"+ str(number_of_crosses) +"/" + str(number_of_crosses)+"_"
    index = 0
    ok = True
    while(ok):
        if os.path.isfile(path+prefix+str(index)+".txt"):
            index +=1
        else:
            ok = False
    return path+prefix+str(index)+".txt"

def get_random_state(init=False):
  velocity = np.zeros(46)

  if init:
    location = np.array([0,0,0.01])
    quaternion = np.array([1,0,0,0])
    angels = np.zeros(40)
    state = np.concatenate([location, quaternion, angels,velocity])

  else:
    z = np.random.uniform(low=-0.2, high=0.2, size=1)
    #z = [0.5]
    location = np.array([0,0,z[0]])
    quaternion = np.random.uniform(low=-2, high=2, size=4)
    angels = np.random.uniform(low=-np.pi/2, high=np.pi/2, size=40)
    state = np.concatenate([location, quaternion, angels,velocity])

  return state

def compare_two_states(predictet_state, gt_state):
    xyz_are_eqaul = np.allclose(predictet_state[:3], gt_state[:3], atol=1e-2)
    quaternion_are_equal = compare_two_states_quaternion(predictet_state[3:7], gt_state[3:7])
    angels_are_eqaul = np.allclose(predictet_state[7:], gt_state[7:], atol=1e-01)
    return xyz_are_eqaul and quaternion_are_equal and angels_are_eqaul

def compare_two_states_quaternion(predictet_state, gt_state):
    predictet = pyt.quaternion_to_matrix(torch.tensor(predictet_state))
    gt = pyt.quaternion_to_matrix(torch.tensor(gt_state))
    equal = torch.allclose(predictet, gt, atol=5e-02) 
    return equal

def get_random_action(min_index, max_index, max_height, min_location, max_location):
    index = np.random.randint(low=min_index, high=max_index+1)
    max_height_value = np.random.uniform(low=0, high=max_height)
    end_location = np.random.uniform(low=min_location, high=max_location, size=2)
    return np.array([index, max_height_value, end_location[0], end_location[1]])

def fix_anachor_state(state, num_of_links=21):
    index = get_xanchor_indexes(num_of_links=num_of_links)
    update_state = []
    for key in index.keys():
        update_state.append(state[index[key]])
    return update_state

def convert_qpos_to_xyz(physics, qpos):
    current_state = copy.deepcopy(physics.get_state())
    set_physics_state(physics, qpos)
    move_center(physics)
    pos = get_position_from_physics(physics)
    set_physics_state(physics, current_state)
    move_center(physics)
    return pos

def convert_qpos_to_xyz_with_move_center(physics, qpos):
    current_state = copy.deepcopy(physics.get_state())
    set_physics_state(physics, qpos)
    move_center(physics)
    pos = get_position_from_physics(physics)
    set_physics_state(physics, current_state)
    move_center(physics)
    return pos

def move_center(physics):
    state = get_current_primitive_state(physics)
    if state[0] != 0:
        temp=1
    state[:2] = 0
    set_physics_state(physics, state)

def get_current_primitive_state(physics):
    """
    return qpos of current physics state
    """
    state = copy.copy(physics.get_state())
    return state
        
def convert_pos_to_topology(pos):
    if not torch.is_tensor(pos):
        pos = torch.tensor(pos)
    pos = pos.detach()
    topology = state2topology(pos, full_topology_representation=True)
    return topology

def acc_score_function_mse(out, trues):
    tensor_predictions = torch.stack(out)
    prediction = torch.mean(tensor_predictions,axis = 0)
    answer = (prediction - trues) * (prediction - trues)
    sum_of_all = torch.sum(answer, dim=1)
    return sum_of_all.tolist()

def uncertainty_score_function_mse(out, batch_size=128):
    uncertainty = []
    for prediction_index in range(batch_size):
        predictions = []
        for index in range(len(out)):
            predictions.append(out[index][prediction_index])
        tensor_predictions = torch.stack(predictions)
        temp_uncertainty = torch.std(tensor_predictions,axis = 0)
        uncertainty.append(temp_uncertainty.sum().item())
    return uncertainty

def uncertainty_score_function_topology(predictions, topology_gt):
    uncertainty = 0
    if len(predictions.shape) != 2:
        raise ValueError('predictions shape is wrong, shape=', len(predictions.shape))
    for prediction_index in range(predictions.shape[0]):
        topology = state2topology(predictions[prediction_index].clone().detach())
        if comperae_two_high_level_states(topology, topology_gt):
            uncertainty +=1 
    return uncertainty/predictions.shape[0]

def uncertainty_score_function_mse_z_only(out, batch_size=128):
    uncertainty = []
    for prediction_index in range(batch_size):
        predictions = []
        for index in range(len(out)):
            predictions.append(out[index][prediction_index])
        for item_index in range(len(predictions)):
            predictions[item_index] = predictions[item_index].view(-1,3)
        tensor_predictions = torch.stack(predictions)
        temp_uncertainty = torch.std(tensor_predictions[:,:,2],axis=0)
        uncertainty.append(temp_uncertainty.sum().item())
    return uncertainty

@torch.no_grad()
def viz_model_output_ensemble(models, dataloader, val_topology_list, output_path ,save=False,\
     batch_size=128, binary=True):
    acc = []
    uncertainty = []
    cnt = {}
    init_index=0
    for _ , (x, y_pos,_) in enumerate(tqdm.tqdm(dataloader)):
        #y_hats = []
        y_hats_pos = []
        y_hats_qpos = []
        if isinstance(x,list):
            x = torch.stack(x)
            x = x.permute(1,0).float()

        if isinstance(y_pos,list):
            y_pos = torch.stack(y_pos)
            y_pos = y_pos.permute(1,0).float()

        if torch.cuda.is_available():
            x = x.cuda()
            #y_pos = y_pos.cuda()

        for model in models:
            y_pos_hat, y_qpos_hat = model(x)
            y_hats_pos.append(y_pos_hat.cpu())
            y_hats_qpos.append(y_qpos_hat.cpu())
        x.cpu()
        acc = acc + conf_mat_score_funcation_topology(out=y_hats_pos, trues=val_topology_list,\
             init_index=init_index)
        init_index += x.shape[0]
        uncertainty = uncertainty + uncertainty_score_function_mse(y_hats_pos, x.shape[0])

    acc_np = torch.tensor(acc)
    acc_np = acc_np.view(-1)
    acc_np = acc_np.tolist()
    uncertainty_np = torch.tensor(uncertainty)
    uncertainty_np = uncertainty_np.view(-1)
    uncertainty_np = uncertainty_np.tolist()
    if binary:
        conf_mat_binary(acc_np, uncertainty_np)
    else:
        conf_mat(acc_np, uncertainty_np)
        
def conf_mat(acc, uncertainty):
    acc_steps = 1
    uncertainty_steps = 10
    max_index = 2
    cfm = np.zeros((max_index, max_index))
    for index in range(len(acc)):
        acc_score = int(acc[index]/acc_steps)
        uncertainty_score = int(uncertainty[index]/uncertainty_steps)
        if acc_score >= max_index or uncertainty_score >= max_index:
            continue
        cfm[acc_score][uncertainty_score] += 1

    classes = [str(i*acc_steps) for i in range(cfm.shape[0])]
    columns = [str(i*uncertainty_steps) for i in range(cfm.shape[1])]

    df_cfm = pd.DataFrame(cfm, index = classes, columns = columns)
    plt.figure(figsize = (25,10))
    cfm_plot = sn.heatmap(df_cfm, annot=True)
    cfm_plot.figure.savefig("outputs/cfm/cfm.png")

def conf_mat_binary(acc, uncertainty):
    pass
    acc_steps = 1
    uncertainty_steps = 0.05
    max_index = 25
    cfm = np.zeros((2, max_index))
    for index in range(len(acc)):
        acc_score = 1 if acc[index] == True else 0 #int(acc[index]/acc_steps)
        uncertainty_score = int(uncertainty[index]/uncertainty_steps)
        if acc_score >= max_index or uncertainty_score >= max_index:
            continue
        cfm[acc_score][uncertainty_score] += 1

    classes = [str(i*acc_steps) for i in range(cfm.shape[0])]
    columns = [str(i*uncertainty_steps) for i in range(cfm.shape[1])]

    df_cfm = pd.DataFrame(cfm, index = classes, columns = columns)
    plt.figure(figsize = (25,10))
    cfm_plot = sn.heatmap(df_cfm, annot=True)
    cfm_plot.figure.savefig("outputs/cfm/cfm_binary.png")

def plot_rope(ax, state, gt=False, save=False):
    state = state.view(-1,3)
    state = state.tolist()
    x = [point[0] for point in state]
    y = [point[1] for point in state]
    if gt:
        ax.plot(x, y, 'ro')
    else:
        ax.plot(x, y, 'bo')

def get_xanchor_indexes(num_of_links: int) -> dict:
    """return dict with index for each position

    Args:
        num_of_links (int): number of links

    Returns:
        dict: index between position and name
    """
    num_links = int(num_of_links)
    index = {}
    for i in range(int(num_links/2)):
        index[i] = (num_links-1-i)*2
    for i in range(int(num_links/2+1), num_links):
        index[i] = (1+(i-int(num_links/2+1)))*2
    return index

def get_joints_indexes(num_of_links: int) -> dict:
    """return dict with index for each joints

    Args:
        num_of_links (int): number of links

    Returns:
        dict: index between joint and name
    """
    #num_links = int(num_of_links)
    index = {}
    for i in range(int(num_of_links/2)):
        for joint_index in [0,1]:
            key = "J"+str(joint_index)+"_"+str(i)
            index[key] = 6 + (num_of_links-1)*2 - i*2 - 1 + joint_index 
    for i in range(int(num_of_links/2+1), num_of_links):
        for joint_index in [0,1]:
            key = "J"+str(joint_index)+"_"+str(i)
            index[key] = 6 + (i-int(num_of_links/2+1))*2 + 1 + joint_index
    return index

def get_number_of_links(physics=None, qpos=None)-> int:
    """return how much links the rope have

    Args:
        physics (dm control): physics of the SIM

    Returns:
        num_of_links(int): num of links
    """
    if physics is not None:
        xanchor = copy.deepcopy(physics.data.xanchor)
        num_of_links = int((len(xanchor)-1)/2+1)
    elif qpos is not None:
        if torch.is_tensor(qpos):
            qpos_length = qpos.shape[1]
            num_of_links = (qpos_length-7)/2+1
    else:
        #not input
        raise
    return int(num_of_links)

def get_position_from_physics(physics) -> np.array:
    """return postion of rope

    Args:
        physics (dm control): physics of the SIM

    Returns:
        position (np.array): rope position relative to joints
    """
    xanchor = copy.deepcopy(physics.data.xanchor)
    num_of_links = get_number_of_links(physics=physics)
    position = np.zeros((num_of_links+1,3))
    state = np.array(fix_anachor_state(xanchor, num_of_links=num_of_links))
    position[1:-1] = state
    start_body_location = physics.named.data.xpos["B0"]
    position[0] = start_body_location-position[1] + start_body_location
    end_body_location = physics.named.data.xpos["B"+str(num_of_links-1)]
    position[-1] = end_body_location-position[num_of_links-1] + end_body_location
    
    return position

def topology_analysis(paths):
    samples = []
    for path in paths:
        files = os.listdir(path)
        for file in tqdm.tqdm(files):
            with open(path+file, "rb") as fp:
                dataset_dict = pickle.load(fp)
                
            if isinstance(dataset_dict, dict):
                raise

            elif isinstance(dataset_dict, list):
                for item in dataset_dict:
                    sample = {}
                    sample["input_number_crosses"] = int((len(item['topology_start'])-2)/2)
                    sample["output_number_crosses"] = int((len(item['topology_end'])-2)/2)
                    sample["input"] = item['topology_start']
                    sample["output"] = item['topology_end']
                    samples.append(copy.deepcopy(sample))
                fp.close()
    return samples

def check_curve_output_oracle(physics, action, state, topology_goal, num_of_links):
    set_physics_state(physics, state)
    physics = execute_action_in_curve_with_mujoco_controller(physics=physics, action=action, num_of_links=num_of_links, return_render=False)
    pos = get_position_from_physics(physics)
    if not torch.is_tensor(pos):
        pos = torch.tensor(pos)
    pos = pos.detach()
    new_topology = state2topology(pos, full_topology_representation=True)
    are_equal = comperae_two_high_level_states(new_topology, topology_goal)
    return are_equal, copy.deepcopy(physics.get_state())

def set_physics_state(physics, state):
    with physics.reset_context():
        physics.set_state(state)

def convert_log_to_video(log_path, number_of_links, output_path, env_path):
    with open(log_path, "rb") as fp:   # Unpickling
        log = pickle.load(fp)
    physics = mujoco.Physics.from_xml_path(env_path)
    physics_reset(physics)

    for action in tqdm.tqdm(log['actions']):
        physics = execute_action_in_curve_with_mujoco_controller(physics, action, num_of_links=15, get_video=True,\
            show_image=False, save_video=True, return_render=False, sample_rate=20, output_path = output_path)  

def convert_topology_to_str(toplogy):
    if isinstance(toplogy, list):
        temp_topology = toplogy
    else:
        temp_topology = toplogy.points

    string = ""
    for item_topology in temp_topology:
        string += str(item_topology)
    return string


def save_pickle(path:str, object_to_save):
    with open(path, "wb") as fp:
        pickle.dump(object_to_save, fp)
    fp.close()

def run_with_limited_time(func, args, kwargs, time):
    """Runs a function with time limit
    :param func: The function to run
    :param args: The functions args, given as tuple
    :param kwargs: The functions keywords, given as dict
    :param time: The time limit in seconds
    :return: True if the function ended successfully. False if it was terminated.
    """
    p = Process(target=func, args=args, kwargs=kwargs)
    p.start()
    p.join(time)
    if p.is_alive():
        p.terminate()
        return False
    return True

def run_with_limited_time_new(func, args=(), kwargs={}, time=1, default=False):
    import signal

    class TimeoutError(Exception):
        pass

    def handler(signum, frame):
        raise TimeoutError()

    # set the timeout handler
    signal.signal(signal.SIGALRM, handler) 
    signal.alarm(time)
    try:
        result = func(*args, **kwargs)
    except TimeoutError as exc:
        result = default
    finally:
        signal.alarm(0)

    return result

def short_topology_state_abstract(topology):
    topology = [i for i in topology if i is not None]
    return topology

def convert_topology_state_to_input_vector(topology):
    topology = short_topology_state_abstract(topology)
    output_length = 300
    output = np.zeros(output_length)
    current_index = 0
    for item in topology:
        if current_index >= output_length:
            break
        if item.over is not None:
            #assert item.over<9
            output[current_index+item.over-1] = 1
        current_index+=8
        if current_index >= output_length:
            break
        if item.sign is not None:
            assert item.sign==1 or item.sign==-1
            if item.sign == 1:
                output[current_index] = 1
            else:
                output[current_index] = -1
        current_index+=1
        if current_index >= output_length:
            break
        if item.under is not None:
            #assert item.under<9
            output[current_index+item.under-1] = 1
        current_index+=8
    return output[:144]

def convert_s2s_to_s2a_dataset(dataset, topology):
    for index in range(len(dataset)):
        item = dataset[index]
        topology_gt = topology[index]
        sample = item[0]
        gt_pos = item[1]
        gt_qpos = item[2]
        output = convert_topology_state_to_input_vector(topology_gt)
        new_output = np.zeros(len(sample) + len(output) - 24) #24 is action
        new_output[:47] = sample[:47]
        new_output[47:113] = sample[71:137]
        new_output[113:] = output[:]
        gt_pos = sample[47:71]
        dataset[index] = [new_output.tolist(), gt_pos]
    return dataset, topology

def edict2dict(edict_obj):
  dict_obj = {}

  for key, vals in edict_obj.items():
    if isinstance(vals, edict):
      dict_obj[key] = edict2dict(vals)
    else:
      dict_obj[key] = vals

  return dict_obj 

######################################archive##########################################################3
"""
def get_start_state_from_x(state):
    state = state.view(-1,6)
    state = state[:,:3]
    return state   

def get_ordered_state(physics):
    points = ["G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10",\
        "G11", "G12", "G13", "G14", "G15", "G16", "G17", "G18", "G19", "G20"]
    new_state = physics.named.data.geom_xpos[points]
    return new_state

def viz_model_output(model, dataloader, output_path, save=False):
    model.train(False)
    for _ , (x, y) in enumerate(dataloader):
        if isinstance(x,list):
            x = torch.stack(x)
            x = x.permute(1,0).float()

        if isinstance(y,list):
            y = torch.stack(y)
            y = y.permute(1,0).float()

        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        y_hat = model(x)
        break
    cnt = 0 
    for index in range(y_hat.shape[0]):
        if(cnt > 200):
            break
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        plot_rope(ax2, y_hat[index], gt=False)
        plot_rope(ax2, y[index], gt=True)
        x_state = get_start_state_from_x(x[index])
        plot_rope(ax1, x_state, gt=False)
        ax1.set_xlim([-1.5, 1.5])
        ax1.set_ylim([-1.5, 1.5])
        ax2.set_xlim([-1.5, 1.5])
        ax2.set_ylim([-1.5, 1.5])
        ax1.set_title("start")
        error = score_function_mse(out=y_hat[index], trues=y.data[index])
        prediction_title = "prediction error = " + str(error)
        ax2.set_title(prediction_title)
        fig.savefig(output_path+"_old_train_"+str(index)+"_with_error.png")
        cnt+=1
        plt.close(fig)
    model.train(True)  

# new methods

def get_observation(physics) -> np.ndarray:
    new_state = get_ordered_state(physics)
    state = torch.tensor(new_state)
    state = state.view(-1)
    return state.cpu().detach().numpy()

"""