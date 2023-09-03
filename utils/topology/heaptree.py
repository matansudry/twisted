import heapq

class Tree(object):
    """ a min heap that also keeps search tree information from RRT.
    Each item in the heapq is a tuple of (priority, id, data),
    where id is the order that samples are generated, to break ties in priority.
    """

    def __init__(self):
        self.V = []
        self.V_count = 0
        self.E = {}

    def add_root(self, root):
        # root is a tuple (priority, 0, data).
        heapq.heappush(self.V, root)
        self.V_count += 1
        self.E[tuple(root[2].flatten())] = None

    def add_leaf(self, parent, child, traj):
        # Parent and child are tuples (priority, id, data).
        heapq.heappush(self.V, child)
        self.V_count += 1
        self.E[tuple(child[2].flatten())] = (tuple(parent[2].flatten()), traj)

    def reconstruct_path(self, leaf):
        # Reconstruct path from root to leaf
        waypoint_path = [leaf]
        actions_path = []
        current = tuple(leaf.flatten())
        while self.E[current] is not None:
            waypoint_path.append(self.E[current][0])
            actions_path.append(self.E[current][1])
            current = self.E[current][0]
        waypoint_path.reverse()
        actions_path.reverse()
        return waypoint_path, actions_path


