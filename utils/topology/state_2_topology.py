from utils.topology.representation import AbstractState, Face
import numpy as np
from shapely.geometry import LineString
import torch

def intersection(first_line_start_point, first_line_end_point, second_line_start_point, second_line_end_point):
    """
    this method getting 4 points, 2 from each line and checking if there is intersection between them
    """
    start_line =  LineString([first_line_start_point, first_line_end_point])
    end_line = LineString([second_line_start_point, second_line_end_point])
    intersection_lines = start_line.intersection(end_line)
    return intersection_lines.bounds

def find_intersections(state):
    #device = state.device.type+":"+str(state.device.index)
    intersections = []
    if isinstance(state, (list, np.ndarray)):
        state = torch.tensor(state)
    new_state = state.view(-1,3)
    for i in range(new_state.shape[0]-1):
        for j in range(i+2, new_state.shape[0]-1):
            intersect = intersection(new_state[i][:2], new_state[i+1][:2], new_state[j][:2], new_state[j+1][:2])
            if intersect:
                intersect_point = torch.Tensor([intersect[0], intersect[1]])
                alpha = np.linalg.norm(intersect_point-new_state[i][:2]) / np.linalg.norm(new_state[i+1][:2]-new_state[i][:2])
                beta = np.linalg.norm(intersect_point-new_state[j][:2]) / np.linalg.norm(new_state[j+1][:2]-new_state[j][:2])
                h_i = alpha*new_state[i+1][2]+(1-alpha)*new_state[i][2]
                h_j = beta*new_state[j+1][2]+(1-beta)*new_state[j][2]
                over = h_i > h_j
                sign = np.cross(new_state[i+1][:2]-new_state[i][:2], new_state[j+1][:2]-new_state[j][:2])
                sign = sign > 0
                if not over:
                    sign = not sign
                over = 1 if over else -1
                sign = 1 if sign else -1
                intersections.append((i,j, over, sign))
    return intersections

def intersect2topology(intersections, update_edges=True, update_faces=True):
    points = [it[0] for it in intersections] + [it[1] for it in intersections]
    points.sort()
    pointDict = {p:i for i,p in enumerate(points)}
    topology = AbstractState()

    for p in points:
        topology.addPoint(1)
    for it in intersections:
        if it[2]==1:
            topology.point_intersect(pointDict[it[0]]+1, pointDict[it[1]]+1, it[3])
        else:
            topology.point_intersect(pointDict[it[1]]+1, pointDict[it[0]]+1, it[3])
        if update_edges:
            if it[3] != it[2]:  # XOR
                i, j = pointDict[it[0]]+1, pointDict[it[1]]+1
            else:
                i, j = pointDict[it[1]]+1, pointDict[it[0]]+1
            topology.link_edges(2*i-2, 2*j-1)
            topology.link_edges(2*j-2, 2*i)
            topology.link_edges(2*i+1, 2*j)
            topology.link_edges(2*j+1, 2*i-1)
    if update_faces:
        visited_edge = [False] * len(topology.edges)
        start_edge_idx = 0
        next_edge = topology.edges[start_edge_idx]
        visited_edge[start_edge_idx] = True
        cnt = 0
        while next_edge.next != start_edge_idx and cnt<10:
            cnt+=1
            visited_edge[next_edge.next] = True
            next_edge = topology.edges[next_edge.next]
        while not all(visited_edge):
            start_edge_idx = visited_edge.index(False)
            new_face_idx = len(topology.faces)
            topology.faces.append(Face(start_edge_idx))
            next_edge = topology.edges[start_edge_idx]
            visited_edge[start_edge_idx]=True
            next_edge.face = new_face_idx
            cnt = 0
            while next_edge.next != start_edge_idx and cnt<10:
                cnt+=1
                visited_edge[next_edge.next] = True
                next_edge = topology.edges[next_edge.next]
                next_edge.face = new_face_idx
    return topology

def state2topology(state, update_edges=True, update_faces=True, full_topology_representation=False):
    num_of_points = state.shape[0]
    intersections = find_intersections(state)
    it_points = [it[0] for it in intersections] + [it[1] for it in intersections]
    new_intersections = intersections
    cnt = 0
    new_state = state.detach().clone()
    while len(set(it_points)) < len(it_points) and cnt<10:
        cnt +=1 
        # deduplicate by breaking segment into smaller ones.
        if not torch.is_tensor(new_state):
            new_state = torch.tensor(new_state)
        new_state = new_state.view(-1,3)
        it_points = sorted(it_points) + [num_of_points]
        segs = []
        current_pt = 0
        current_rep_count = 0
        for it in it_points:
            if it > current_pt:
                if current_rep_count < 2:
                    segs.append(new_state[current_pt:it])
                else:
                    alpha = np.linspace(0.0,1.0,current_rep_count*2+1)
                    points = new_state[current_pt]*(1-alpha[0:-1,np.newaxis]) + new_state[current_pt+1]*alpha[0:-1,np.newaxis]
                    segs.append(points)
                    segs.append(new_state[current_pt+1:it])
                current_pt = it
                current_rep_count = 1
            else:
                current_rep_count +=1

        new_state = np.concatenate(segs, axis=0)
        new_intersections = find_intersections(new_state)
        it_points = [it[0] for it in new_intersections] + [it[1] for it in new_intersections]

    topology = intersect2topology(new_intersections, update_edges, update_faces)
    if full_topology_representation:
        return topology
    else:
        return topology.points #old - topology, intersections
