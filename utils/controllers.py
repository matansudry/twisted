import numpy as np

class Basic_Controller():
    def __init__(self):
        pass


class Path_Controller(Basic_Controller):
    def __init__(self, k):
        super(Basic_Controller, self).__init__()
        self.index = 1
        self.k = k

    def get_force_from_spline_and_state(self, state, trjectroy):
        """
        this method will get the current state and the trejctory path and return force on object
        """
        goal = trjectroy[self.index]
        force = self.k * (np.array(goal) - np.array(state))
        force[2] = 200*force[2]
        self.increase_index()
        return force
        
    def increase_index(self):
        self.index +=1

    def reset(self):
        self.index = 1