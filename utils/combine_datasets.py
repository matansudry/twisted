#packges
import numpy as np
import argparse
import torch
import cv2
import json
import tqdm
import argparse
import os

#files
from utils.general_utils import create_spline_from_points
from dm_control import mujoco

"""
class Basic_Controller():
    def __init__(self):
        pass

class Path_Controller(Basic_Controller):
    def __init__(self, k):
        super(Basic_Controller, self).__init__()
        self.index = 1
        self.k = k

    def get_force_from_spline_and_state(self, state, trjectroy):
        goal = trjectroy[self.index]
        force = self.k * (np.array(goal) - np.array(state))
        force[2] = 200*force[2]
        self.increase_index()
        return force
        
    def increase_index(self):
        self.index +=1

    def reset(self):
        self.index = 1

def get_ordered_state(physics):
    points = ["G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10",\
        "G11", "G12", "G13", "G14", "G15", "G16", "G17", "G18", "G19", "G20"]
    new_state = physics.named.data.geom_xpos[points]
    return new_state

def get_observation(physics) -> np.ndarray:
    new_state = get_ordered_state(physics)
    state = torch.tensor(new_state)
    state = state.view(-1)
    return state.cpu().detach().numpy()

def get_random_action():
    index = np.random.randint(low=0, high=20)
    max_height = np.random.uniform(low=0, high=0.3)
    end_location = np.random.uniform(low=-0.5, high=0.5, size=2)
    return np.array([index,max_height, end_location[0], end_location[1]])

def execute_action_in_curve(physics, action, get_image=False, get_video=False, x_step_size=0.0001, k_value=0.1):
    index = int(action[0])
    joint = "G"+str(int(index))
    start_point = physics.named.data.geom_xpos[joint]
    end_point= [action[2],action[3], 0]
    max_height = action[1]

    spline = create_spline_from_points(start_point=start_point, max_height=max_height, end_point=end_point,\
       x_step_size=x_step_size)

    controller = Path_Controller(k_value)
    
    images = None
    if get_video:
      images = []

    for _ in range(len(spline)-1):
      state = physics.named.data.geom_xpos[joint]
      force = controller.get_force_from_spline_and_state(state, spline)
      action = [force[0], force[1], force[2], 0., 0., 0.]
      physics.named.data.xfrc_applied[index+1] = action
      physics.step()
      if get_image or get_video:
        pixels = physics.render()
        if get_video:
          images.append(pixels)
        if get_image:
          cv2.imshow("image", pixels)
          cv2.waitKey(1)

    for _ in range(20):
      action = [0., 0., 0., 0., 0., 0.]
      physics.named.data.xfrc_applied[index+1] = action
      if get_image:
        pixels = physics.render()
        if get_video:
          images.append(pixels)
        cv2.imshow("image", pixels)
        cv2.waitKey(1)
    return images

def main():
    full_dataset = []
    output_location = "datasets/"
    for index in tqdm.tqdm(range(6000,11000)):
        f = open(output_location+"run"+str(index)+".json")
        data = json.load(f)
        for item in data.values():
            full_dataset.append(item)
    import pickle
    with open(output_location+"full_dataset_3.txt", "wb") as fp:
        pickle.dump(full_dataset, fp)


def main_new():
    import pickle
    output_location = "datasets/"
    with open(output_location+"full_dataset_combine_1_2_with_topology.txt", "rb") as fp:   # Unpickling
        dataset = pickle.load(fp)
        with open(output_location+"full_dataset_3.txt", "rb") as fp_2:   # Unpickling
            new_dataset = pickle.load(fp_2)
            for item in tqdm.tqdm(new_dataset):
                dataset.append(item)
            with open(output_location+"full_dataset_combine_1_2_with_topology_3.txt", "wb") as fp:
                pickle.dump(dataset, fp)

if __name__ == '__main__':
    raise #[MS] the code is not updated
    main_new()
"""







