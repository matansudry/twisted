import torch

from system_flow.low_level_class.base_low_level import LowLevelPlanner

class RandomPlanner(LowLevelPlanner):
    def __init__(self, cfg, config_length):
        super(RandomPlanner, self).__init__(cfg, config_length)
        #Random action
        self.low_index = 0
        self.high_index = self.num_of_links
        self.low_height = self.cfg['STATE2ACTION_PARMS']['output_ranges'].height[0]
        self.high_height = self.cfg['STATE2ACTION_PARMS']['output_ranges'].height[1]
        self.low_x = self.cfg['STATE2ACTION_PARMS']['output_ranges'].x[0]
        self.high_x = self.cfg['STATE2ACTION_PARMS']['output_ranges'].x[1]
        self.low_y = self.cfg['STATE2ACTION_PARMS']['output_ranges'].y[0]
        self.high_y = self.cfg['STATE2ACTION_PARMS']['output_ranges'].y[1]
        self.batch_size = self.cfg['batch_size']

    def generate_action(self, configuration, target_topology_state, physics, playground_physics):
        int_part = torch.randint(self.low_index, self.high_index,(self.batch_size,1))
        continues_part = torch.rand(self.batch_size,3)
        #height
        continues_part[:,0] *= self.high_height - self.low_height
        continues_part[:,0] += self.low_height

        #x,y part
        continues_part[:,1] *= self.high_x - self.low_x
        continues_part[:,1] += self.low_x
        continues_part[:,2] *= self.high_y - self.low_y
        continues_part[:,2] += self.low_y

        #concat
        batch = torch.cat((int_part,continues_part), 1)
        return batch