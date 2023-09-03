import numpy as np
import matplotlib.pyplot as plt
from dynamics_inference.dynamic_models import physbam_3d
from povray_render.sample_spline import sample_b_spline, sample_equdistance
from representation import AbstractState
from state_2_topology import state2topology
from BFS import bfs
import pickle
import pdb


#with open('fitted_gaussians_3d.pkl', 'rb') as f:
#    fitted = pickle.load(f)
#gaussian=fitted['idx:0 left:1 move:R1 sign:1']
#gaussian_mean, gaussian_std, gaussian_cov = gaussian[0], gaussian[1], gaussian[2]
#load_gmm_action_samples = np.load('gmm_samples.npy')

#start_state = np.zeros((64,3))
#start_state[:,0] = np.linspace(-0.5, 0.5, 64)

with open('fitted_gaussian_3d_with_1_intersection.pkl', 'rb') as f:
    fitted = pickle.load(f)
gaussian=fitted['move:cross over_idx:2 sign:1 under_idx:1']
gaussian_mean, gaussian_std, gaussian_cov = gaussian[0], gaussian[1], gaussian[2]

start_state = np.loadtxt('start_state_1_intersection.txt')

batch_actions = []
traj_params = []
for i in range(200):
    #action = np.random.normal(loc=gaussian_mean, scale=gaussian_std)
    action = np.random.multivariate_normal(gaussian_mean, gaussian_cov)
    #action = load_gmm_action_samples[i]
    action_node = int(action[0]*63)
    action_node = np.clip(action_node, 0, 63)
    action_traj = action[1:5]
    height = action[5]
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

dynamic_inference = physbam_3d(' -friction 0.1 -stiffen_linear 0.232 -stiffen_bending 0.6412 -self_friction 0.4649')

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

pdb.set_trace()

