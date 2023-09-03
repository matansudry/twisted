import sys
sys.path.append(".")


import numpy as np
import pytorch3d.transforms as pyt

from utils.general_utils import *
from state2state_flow.s2s_utils.dataset_utils import *

from dm_control import mujoco


if __name__ == '__main__':
    path = "exp/dicrete_results"
    files = os.listdir(path)
    # a = answer['continues_discrete_equal']=False, answer['continues_goal_equal']=False
    layer_3 = {
        True: 0,
        False: 0
    }
    layer_2 = {
        True: copy.deepcopy(layer_3),
        False: copy.deepcopy(layer_3)
    }
    layer_1 = {
        True: copy.deepcopy(layer_2),
        False: copy.deepcopy(layer_2)
    }

    for file in tqdm.tqdm(files):
        answer = load_pickle(path+"/"+file)
        layer_1[answer['continues_discrete_equal']][answer['continues_goal_equal']][answer['discrete_goal_equal']] +=1
    print("continues_discrete_equal, continues_goal_equal, discrete_goal_equal")
    print_order = [
        [False,False,False],
        [False,False,True],
        [False,True,False],
        [True,False,False],
        [True,True,True]
    ]

    sum = 0
    for item in print_order:
        sum += layer_1[item[0]][item[1]][item[2]]

    for item in print_order:
        cnt = layer_1[item[0]][item[1]][item[2]]
        print("cnt =", cnt/sum)
    temp=1