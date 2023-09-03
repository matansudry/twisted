from povray_render.sample_spline import *
from povray_render.lift_3d import intersection
from dynamics_inference.dynamic_models import *
import numpy as np
import matplotlib.pyplot as plt
import pdb

def find_intersections(state):
    intersections = []
    for i in range(state.shape[0]-1):
        for j in range(i+2, state.shape[0]-1):
            intersect = intersection(state[i], state[i+1], state[j], state[j+1])
            if intersect is not None:
                intersections.append((i,j))
    return intersections

def intersect2topology(intersections):
    points = [it[0] for it in intersections] + [it[1] for it in intersections]
    points.sort()
    pointDict = {p:i for i,p in enumerate(points)}
    topology = []
    for p in points:
        for it in intersections:
            if p==it[0]:
                other=it[1]
                break
            elif p==it[1]:
                other=it[0]
                break
        topology.append(pointDict[other])
    return topology, points

def reward(state, requested_intersections, requested_loops):
    """ requested_loops is a list of loops, each loop is represented by a list of segments.
    Segments are represented as (p1,p2) where p1 and p2 are indexed sequentially from 0.
    p1 and p2 are not node index on the rope.
    """
    intersections = find_intersections(state)
    topology, points = intersect2topology(intersections)
    requested_topology, requested_points = intersect2topology(requested_intersections)
    if topology == requested_topology:
        reward = 100
        for p, rp in zip(points, requested_points):
            reward += 25-np.abs(p-rp)

        def cross(v1,v2):
            return v1[0]*v2[1]-v1[1]*v2[0]
        areas = []
        for loop in requested_loops:
            area = 0
            for p1,p2 in loop:
                if p2 > p1:
                    for k in range(points[p1], points[p2]):
                        area += cross(state[k], state[k+1])
                else:
                    for k in range(points[p1], points[p2], -1):
                        area += cross(state[k], state[k-1])
            print('area: ', np.abs(area))
            areas.append(np.abs(area))
        softmin = np.log(float(len(areas)))-np.log(np.sum(np.exp(areas)))
        return reward+softmin*20
    else:
        return 0


class CEM(object):
    def __init__(self, batch_size, inv_temperature, num_iterations,
                 dynamic_type, param_dict, fixed_argument):
        # param_dict has key:(mean, std) entries,
        # where key is the argument name in physbam.
        # fixed_arguments is a string of fixed physbam arguments.
        self.batch_size = batch_size
        self.inv_temperature = inv_temperature
        self.num_iterations = num_iterations
        self.dynamic_type = dynamic_type
        self.param_dict = param_dict
        self.fixed_argument = fixed_argument

        self.current_state =  np.zeros((64,2))
        self.current_state[:,0] = np.linspace(0,1,64)
        self.requested_intersections = [(10, 40)]
        self.requested_loops = [[(0,1)]]
        self.action_node = 0
        assert(dynamic_type=='physbam_2d')
        self.dynamics = physbam_2d()

    def sample(self):
        params = dict()
        for key,val in self.param_dict.items():
            params[key] = np.random.normal(
                              loc=val[0], scale=val[1], size=self.batch_size)
        self.params = [{key:val[i] for key,val in params.items()} for i in range(self.batch_size)]

    def evaluate(self):
        self.sample_score = []
        batch_actions = []
        for param in self.params:
            sample_knots = [tuple(self.current_state[self.action_node]),
                            tuple(self.current_state[self.action_node]),
                            (param['mx'], param['my']),
                            (param['dx'], param['dy']),
                            (param['dx'], param['dy'])]
            samples = sample_cubic_spline(sample_knots)
            samples = sample_equdistance(samples, None, seg_length=0.02)
            actions = [(self.action_node, samples[:,i]-samples[:,i-1]) for i in range(1,samples.shape[1])]
            batch_actions.append(actions)
        final_states = self.dynamics.execute_batch(self.current_state, batch_actions)
        for state in final_states:
            self.sample_score.append(reward(state, self.requested_intersections, self.requested_loops))

    def update(self):
        weights = [np.exp(score*self.inv_temperature) for score in self.sample_score]
        for key in self.param_dict:
            vals = [param[key] for param in self.params]
            mean = np.average(vals, weights=weights)
            vals = [(v-mean)**2 for v in vals]
            std = np.sqrt(np.average(vals, weights=weights))
            self.param_dict[key] = (mean, std)

    def run(self):
        for _ in range(self.num_iterations):
            print("Argument: (mean, std)\n")
            for key, val in self.param_dict.items():
                print("%s: (%f, %f)\n" % (key, val[0], val[1]))
            self.sample()
            self.evaluate()
            keys = self.param_dict.keys() # to ensure order of keys / vals.
            print(", ".join(keys) + ", Score:")
            for p,s in zip(self.params, self.sample_score):
                print([p[k] for k in keys] + [s])
            self.update()
        idx = np.argmax(self.sample_score)
        return self.params[idx]

if __name__ == "__main__":
    requested_intersections = [(10,40)]
    requested_loops = [[(0,1)]]
    action_node = 0
    init_dict = {'mx': (0.35,0.5),
                 'my': (0, 0.5),
                 'dx': (0.65, 0.5),
                 'dy': (0, 0.5)}
    identifier = CEM(batch_size=64, inv_temperature=0.2, num_iterations=5,
                     dynamic_type = 'physbam_2d', param_dict = init_dict,
                     fixed_argument= '')
    identifier.requested_intersections = requested_intersections
    identifier.requested_loops = requested_loops
    identifier.action_node = action_node
    param = identifier.run()
    print("ID: ", param)
    sample_knots = [(0,0), (0,0), (param['mx'], param['my']),
                    (param['dx'], param['dy']), (param['dx'], param['dy'])]
    samples = sample_cubic_spline(sample_knots)
    samples = sample_equdistance(samples, None, seg_length=0.02)

    actions = [(action_node, samples[:,i]-samples[:,i-1]) for i in range(1,samples.shape[1])]
    # rollout and save nominal trajectory
    state = identifier.current_state
    states = [state]
    for action in actions:
        state = identifier.dynamics.execute(state, [action])
        states.append(state)
    final_state = state
    np.save('nominal_traj_1.state', np.array(states))
    with open('nominal_traj_1.action', 'w') as f:
        for action in actions:
            f.write('%d %f %f\n'%(action[0], action[1][0], action[1][1]))

    plt.plot(final_state[:,0], final_state[:,1])
    plt.axis('equal')
    plt.show()

    pdb.set_trace()
    requested_intersections = [(20,55), (10,40)]
    requested_loops = [[(0,1), (1,2)], [(2,3), (1,0)]]
    action_node = 63
    init_dict = {'mx': (0.5*final_state[63,0]+0.5*final_state[20,0], 0.2),
                 'my': (0.5*final_state[63,1]+0.5*final_state[20,1], 0.2),
                 'dx': (final_state[20,0], 0.1),
                 'dy': (final_state[20,1], 0.1)}
    identifier = CEM(batch_size=64, inv_temperature=0.2, num_iterations=5,
                     dynamic_type = 'physbam_2d', param_dict = init_dict,
                     fixed_argument= '')
    identifier.current_state = final_state
    identifier.requested_intersections = requested_intersections
    identifier.requested_loops = requested_loops
    identifier.action_node = action_node
    param = identifier.run()
    print("ID: ", param)
    sample_knots = [tuple(final_state[action_node]), tuple(final_state[action_node]),
                    (param['mx'], param['my']),
                    (param['dx'], param['dy']), (param['dx'], param['dy'])]
    samples = sample_cubic_spline(sample_knots)
    samples = sample_equdistance(samples, None, seg_length=0.02)
    actions = [(action_node, samples[:,i]-samples[:,i-1]) for i in range(1,samples.shape[1])]
    # rollout and save nominal trajectory
    state = identifier.current_state
    states = [state]
    for action in actions:
        state = identifier.dynamics.execute(state, [action])
        states.append(state)
    final_state = state
    np.save('nominal_traj_2.state', np.array(states))
    with open('nominal_traj_2.action', 'w') as f:
        for action in actions:
            f.write('%d %f %f\n'%(action[0], action[1][0], action[1][1]))

    plt.plot(final_state[:,0], final_state[:,1])
    plt.axis('equal')
    plt.show()

    pdb.set_trace()
    requested_intersections = [(20,55), (10,40), (37,60)]
    requested_loops = [[(0,1), (4,5), (2,3)], [(1,2), (5,4)], [(3,4), (1,0)]]
    action_node = 37
    init_dict = {'mx': (0.5*final_state[37,0]+0.5*final_state[60,0], 0.2),
                 'my': (0.5*final_state[37,1]+0.5*final_state[60,1], 0.2),
                 'dx': (final_state[60,0], 0.1),
                 'dy': (final_state[60,1], 0.1)}
    identifier = CEM(batch_size=64, inv_temperature=0.2, num_iterations=5,
                     dynamic_type = 'physbam_2d', param_dict = init_dict,
                     fixed_argument= '')
    identifier.current_state = final_state
    identifier.requested_intersections = requested_intersections
    identifier.requested_loops = requested_loops
    identifier.action_node = action_node
    param = identifier.run()
    print("ID: ", param)
    sample_knots = [tuple(final_state[action_node]), tuple(final_state[action_node]),
                    (param['mx'], param['my']),
                    (param['dx'], param['dy']), (param['dx'], param['dy'])]
    samples = sample_cubic_spline(sample_knots)
    samples = sample_equdistance(samples, None, seg_length=0.02)
    actions = [(action_node, samples[:,i]-samples[:,i-1]) for i in range(1,samples.shape[1])]
    # rollout and save nominal trajectory
    state = identifier.current_state
    states = [state]
    for action in actions:
        state = identifier.dynamics.execute(state, [action])
        states.append(state)
    final_state = state
    np.save('nominal_traj_3.state', np.array(states))
    with open('nominal_traj_3.action', 'w') as f:
        for action in actions:
            f.write('%d %f %f\n'%(action[0], action[1][0], action[1][1]))
    plt.plot(final_state[:,0], final_state[:,1])
    plt.axis('equal')
    plt.show()

