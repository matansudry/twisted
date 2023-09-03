import networkx as nx
from utils.general_utils import comperae_two_high_level_states
import numpy as np

class GraphManager():
    def __init__(self, config_length):
        self.graph = nx.DiGraph()
        self.id = 0
        self.config_length = config_length

    def add_edge(self, from_id, to_id, attribute):
        self.graph.add_edges_from([
            (from_id, to_id, attribute)
        ])

    def add_node(self, topology_state, configuration, parent_id, action):
        node_attribute_dict = {
            "topology_state": topology_state,
            "configuration": configuration,
            "success": "unknown",
            "number_of_visits": 0
        }
        self.graph.add_nodes_from([
            (self.id, node_attribute_dict),
        ])
        if parent_id is not None:
            self.add_edge(parent_id, self.id, {"action": action})
        else:
            self.set_success_on_node(self.id, "root")
        self.id += 1

        return self.id -1

    def get_parent_id(self, configuration):
        for node in self.graph.nodes():
            node_attributes = self.graph.nodes._nodes[node]
            if (np.allclose(np.array(node_attributes['configuration']), np.array(configuration), atol=1e-5)):
                return node

    def get_all_states_with_topology_state(self, topology):
        states = []
        number_of_visits = []
        for node in self.graph.nodes():
            node_attributes = self.graph.nodes._nodes[node]
            if comperae_two_high_level_states(node_attributes['topology_state'], topology):
                states.append(node_attributes['configuration'])
                number_of_visits.append(node_attributes['number_of_visits'])
        return states, number_of_visits

    def get_all_states(self):
        """
        get all configurations in the graph
        Returns:
            states(list[configurations]): list with all the configurations
        """
        states = []
        for node in self.graph.nodes():
            node_attributes = self.graph.nodes._nodes[node]
            states.append(node_attributes['configuration'])
        return states

    def update_number_of_visits(self, config):
        for node in self.graph.nodes():
            node_attributes = self.graph.nodes._nodes[node]
            if np.array_equal(node_attributes['configuration'],config):
                self.graph.nodes._nodes[node]['number_of_visits']+=1
                break
      

    def set_success_on_node(self, node_id, success):
        if self.graph.nodes._nodes[node_id]["success"] != "root":
            self.graph.nodes._nodes[node_id]["success"] = success

    def check_topology_goal(self, topology):
        for node in self.graph.nodes():
            node_attributes = self.graph.nodes._nodes[node]
            if comperae_two_high_level_states(node_attributes['topology_state'], topology):
                return True
        return False

    def get_all_parents_of_topology_state(self, topology):
        states = []
        for node in self.graph.nodes():
            node_attributes = self.graph.nodes._nodes[node]
            if comperae_two_high_level_states(node_attributes['topology_state'], topology):
                for out_node in self.graph.nodes():
                    if node in self.graph_manager.graph.edges._adjdict[out_node].keys() and\
                        self.graph.nodes._nodes[out_node]["success"] != "no":
                        states.append(self.graph.nodes._nodes[out_node]["configuration"])
        return states

    def get_all_nodes_ids_from_topology_state(self, topology):
        states = []
        for node in self.graph.nodes():
            node_attributes = self.graph.nodes._nodes[node]
            if node_attributes['success'] != "no" and \
                comperae_two_high_level_states(node_attributes['topology_state'], topology):
                states.append(node)
        return states

    def trajctory_extractor(self, goal):
        goal_ids = self.get_all_nodes_ids_from_topology_state(goal)
        path = []
        for temp_goal in goal_ids:
            try:
                path = nx.shortest_path(self.graph, source=0, target=temp_goal)
                break
            except:
                continue
        return path