import sys
sys.path.append(".")

import torch
import numpy as np

from system_flow.low_level_class.random_planner import RandomPlanner
from system_flow.high_level_class.high_level import HighLevelPlanner

from utils.general_utils import set_physics_state, get_current_primitive_state, move_center,\
    convert_qpos_to_xyz_with_move_center, convert_pos_to_topology, comperae_two_high_level_states

with torch.no_grad():
    class RandomHighLevelPlanner(HighLevelPlanner):
        def init_low_level(self, low_cfg):
            self.low_planner = RandomPlanner(low_cfg, self.config_length)

        def _get_random_configuration(self):
            configurations = self.graph_manager.get_all_states()

            number_of_configurations = len(configurations)
            prob = np.ones(number_of_configurations) / number_of_configurations
            options = np.arange(0, len(configurations), 1, dtype=int)
            configuration_index = np.random.choice(options, 1, p=prob)
            configuration = configurations[configuration_index[0]]

            return configuration

        def run(self):
            self.low_planner.batch_size = 1
            while True:
                #generate random action
                batch = self.low_planner.generate_action(configuration=None, target_topology_state=None,\
                     physics=None, playground_physics=None)
                action = batch[0]

                #select configuration
                configuration = self._get_random_configuration()
                
                #execute action
                set_physics_state(self.physics, configuration) 
                try: 
                    self.physics = self.execute_action_in_curve(action, self.physics)
                except:
                    continue

                #calculate topology state
                move_center(self.physics)
                new_primitive_state = get_current_primitive_state(self.physics)
                new_pos_state = convert_qpos_to_xyz_with_move_center(self.playground_physics, new_primitive_state)
                current_topology_state = convert_pos_to_topology(new_pos_state)

                #check if goal is reachd
                if comperae_two_high_level_states(current_topology_state, self.topology_goal):
                    print ("found plan")
                    return True

                #add new node
                parent_id = self.graph_manager.get_parent_id(configuration)
                _ = self.graph_manager.add_node(current_topology_state, new_primitive_state, parent_id,\
                                action)
