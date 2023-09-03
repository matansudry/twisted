import numpy as np
import torch

from system_flow.low_level_class.base_low_level import LowLevelPlanner
from state2action_flow.s2a_models.autoregressive_stochastic_network import autoregressive_stochastic_s2a_netowrk
from utils.general_utils import set_physics_state, get_position_from_physics, convert_topology_state_to_input_vector

class S2APlanner(LowLevelPlanner):
    def __init__(self, cfg, config_length):
        super(S2APlanner, self).__init__(cfg, config_length)
        
        #init varibels
        self.s2a_path = self.cfg["STATE2ACTION_PARMS"]["path"]
        self.s2a_input_size = self.cfg["STATE2ACTION_PARMS"]["input_size"]
        self.s2a_output_size = self.cfg["STATE2ACTION_PARMS"]["output_size"]
        self.batch_size = self.cfg["batch_size"]

        #init and load  NN
        self.init_state2action_nn()
        self.load_state2action_nn()

    def init_state2action_nn(self):
            height = self.cfg["STATE2ACTION_PARMS"]["output_ranges"]["height"]
            x = self.cfg["STATE2ACTION_PARMS"]["output_ranges"]["x"]
            y = self.cfg["STATE2ACTION_PARMS"]["output_ranges"]["y"]
            output_ranges = {
                "height": np.array(height),
                "x": np.array(x),
                "y": np.array(y),
            }
            self.s2a_model = autoregressive_stochastic_s2a_netowrk(
                input_size=self.s2a_input_size,
                output_size=self.s2a_output_size,
                output_ranges=output_ranges,
                dropout=0
                )
    
    def load_state2action_nn(self):
        init = torch.load(self.s2a_path)
        model_state = init["model_state"]
        update_model = {}
        for i in model_state:
            new_i = i.replace('module.', '')
            update_model[new_i] = model_state[i]
        self.s2a_model.load_state_dict(update_model)
        self.s2a_model = self.s2a_model.cuda()

    def generate_action(self, configuration, target_topology_state, physics, playground_physics):
        batch_size = self.batch_size
        set_physics_state(playground_physics, configuration)
        pos = get_position_from_physics(playground_physics)
        pos = np.reshape(pos, -1)
        topology_vector = convert_topology_state_to_input_vector(target_topology_state.points)
        x = np.zeros(self.s2a_input_size)
        x[:47] = configuration[:47]
        x[47:113] = pos[:]
        x[113:] = topology_vector[:]
        #inference
        x = list(x)
        x = torch.tensor(x)

        x = torch.unsqueeze(x, 0)
        x = x.repeat(batch_size,1)

        if torch.cuda.is_available():
            x = x.cuda()

        actions_index, params = self.s2a_model.get_prediction(x.float())
        action_index, height, x_pos, y_pos = self.s2a_model._output_to_sample(actions_index, params)
        
        #get actions
        action_index = action_index.view(-1,1)
        height = height.view(-1,1)
        x_pos = x_pos.view(-1,1)
        y_pos = y_pos.view(-1,1)
        output = torch.cat((action_index, height, x_pos, y_pos),1)
        
        return output