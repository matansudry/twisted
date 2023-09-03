import numpy as np
import matplotlib.pyplot as plt
#from dynamics_inference.dynamic_models import physbam_3d
from povray_render.sample_spline import sample_b_spline, sample_equdistance
from representation import AbstractState
from state_2_topology import state2topology
from BFS import bfs
import pickle
import pdb


start_state = np.zeros((64,3))
start_state[:,0] = np.linspace(-0.5, 0.5, 64)

batch_actions = []
traj_params = []
for _ in range(3000):
    action_node = np.random.choice(list(range(4,60,5)))
    action_traj = np.random.uniform(-0.5,0.5,4)
    height = np.random.uniform(0.02, 0.2)
    knots = [start_state[action_node][:2], start_state[action_node][:2],
             action_traj[0:2],
             action_traj[2:4], action_traj[2:4]]
    traj = sample_b_spline(knots)
    traj = sample_equdistance(traj, None, seg_length=0.01).transpose()
    print(traj.shape)

    # generating 3D trajectory
    traj_height = np.arange(traj.shape[0]) * 0.01
    traj_height = np.minimum(traj_height, traj_height[::-1])
    traj_height = np.minimum(traj_height, height)
    height = np.minimum(height, np.amax(traj_height))
    traj = np.concatenate([traj, traj_height[:,np.newaxis]], axis=-1)

    moves = traj[1:]-traj[:-1]
    actions = [(action_node, m) for m in moves]
    batch_actions.append(actions)
    traj_params.append((action_node/63, action_traj, height))

dynamic_inference = physbam_3d(' -friction 0.1 -stiffen_linear 1.223 -stiffen_bending 1.218')

start_state_raw = state_to_mesh(start_state_raw)
start_state_raw = start_state_raw.dot(np.array([[1,0,0],[0,0,1],[0,-1,0]]))
all_states_raw = dynamic_inference.execute_batch(start_state_raw, batch_actions, return_traj=False, reset_spring=True)
traj_params = [tj for st,tj in zip(all_states_raw, traj_params) if st is not None]
states = [0.5*(st[:64]+st[64:]) for st in all_states_raw if st is not None]

start_abstract_state, _ = state2topology(start_state, update_edges=True, update_faces=True)
end_abstract_state = [state2topology(state, update_edges=True, update_faces=False) for state in states]

dataset_abstract_actions = []
dataset_traj_params = []
for i, (astate, intersection) in enumerate(end_abstract_state):
    intersect_points = [i[0] for i in intersection] + [i[1] for i in intersection]
    if len(set(intersect_points)) < len(intersect_points):
        continue
    _, path_action = bfs(start_abstract_state, astate, max_depth=1)
    if len(path_action)==1:
        dataset_abstract_actions.append(path_action[0])
        dataset_traj_params.append(traj_params[i])

def hash_dict(abstract_action):
    tokens = [k+':'+str(v) for k,v in abstract_action.items()]
    return ' '.join(sorted(tokens))

dataset = {}
for abstract_action, traj_param in zip(dataset_abstract_actions, dataset_traj_params):
    abstract_action_str = hash_dict(abstract_action)
    classified = False
    for ac_str in dataset:
        if abstract_action_str == ac_str:
            dataset[ac_str].append(traj_param)
            classified = True
    if not classified:
        dataset[abstract_action_str] = [traj_param]

with open('gen_data_3d.pkl', 'wb') as f:
    pickle.dump(dataset, f)

# TODO try on 1intersection-> more
