import numpy as np
import torch
from collections import namedtuple
from gym.spaces import Box

from system_flow.low_level_class.base_low_level import LowLevelPlanner
from utils.general_utils import *
from rl_baseline.sac_algorithm import SACAlgorithm
from rl_baseline.mujoco_env import MujocoKnotTyingEnv


class RLPlanner(LowLevelPlanner):
    def __init__(self, cfg, config_length):
        super(RLPlanner, self).__init__(cfg, config_length)

        self.rl_config = cfg["RL"]
        self.config = load_pickle(os.path.join(self.rl_config["path"], "config.txt"))
        self.config = namedtuple('struct', self.config.keys())(*self.config.values())
        self.env = MujocoKnotTyingEnv(**self.config.env_params)
        self.agent = SACAlgorithm(self.config, self.env)

        self._load_checkpoint()
    
    def _load_checkpoint(self):
        path = os.path.join(self.rl_config["path"], "best_model")
        init = torch.load(path)
        self.agent.load_state_dict(init, strict=False)

    def generate_action(self, configuration, target_topology_state, physics, playground_physics):
        batch_size = self.batch_size

        state_pos = convert_qpos_to_xyz_with_move_center(playground_physics, configuration)
        state_topology = convert_pos_to_topology(state_pos)
        achieved_goal = convert_topology_state_to_input_vector(state_topology.points)

        sample = {
            "observation": np.float32(configuration[2:47]),
            "desired_goal": np.float32(convert_topology_state_to_input_vector(target_topology_state.points)),
            "achieved_goal": np.float32(achieved_goal)
        }

        batch = []
        for _ in range(batch_size):
            batch.append(sample)

        actions = self.agent.predict_action(batch, deterministic=False)
        actions = self._update_action_to_index(actions)

        torch_action = torch.tensor(actions)

        return torch_action

    def _update_action_to_index(self, actions):
        actions[:,0] = np.floor(actions[:,0] * 21)
        return actions 