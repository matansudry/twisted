from operator import truediv
import random, copy
import pdb
from utils.topology.representation import AbstractState
import time
import numpy as np
from operator import itemgetter
from utils.general_utils import convert_topology_to_str, load_pickle

def generate_next(state):
    # cross action
    for over_idx in range(0, state.pts+1):
        for under_idx in range(0, state.pts+1):
            for sign in [-1, 1]:
                new_state = copy.deepcopy(state)
                success = new_state.cross(over_idx, under_idx, sign)
                if success:
                    action = {'move':'cross', 'over_idx':over_idx, 'under_idx':under_idx, 'sign':sign}
                    yield new_state, action

    # R1 action
    for idx in range(0, state.pts+1):
        for sign in [1, -1]:
            for left in [1, -1]:
                new_state = copy.deepcopy(state)
                success = new_state.Reide1(idx, sign, left)
                if success:
                    action = {'move':'R1', 'idx':idx, 'sign':sign, 'left':left}
                    yield new_state, action

    # R2 action
    for over_idx in range(0, state.pts+1):
        for under_idx in range(0, state.pts+1):
            for left in [-1, 1]:
                for obu in [-1, 1]:
                    new_state = copy.deepcopy(state)
                    success = new_state.Reide2(over_idx, under_idx, left=left, over_before_under=obu)
                    if success:
                        action = {'move':'R2', 'over_idx':over_idx, 'under_idx':under_idx, 'left':left, 'over_before_under':obu}
                        yield new_state, action

def bfs(start, goal, max_depth=None):
    # using max_depth to limit search. mainly used for data generation.
    visited, parents, actions = [start], [0], [{}]
    depth_index = [0]
    if start == goal:
        return [start], []
    head = 0
    while head < len(visited):
        state = visited[head]
        if head>depth_index[-1]:
            depth_index.append(len(visited)-1)
        if max_depth is not None and len(depth_index) > max_depth:
            return [],[]
        for new_state, action in generate_next(state):
            append = True
            for visited_state in visited:
                if new_state == visited_state:
                    append = False
            if append:
                visited.append(new_state)
                parents.append(head)
                actions.append(action)
            if new_state == goal:
                break
        head += 1
        if visited[-1] == goal:
            head = len(visited)-1
            break
    # backtrack to find solution
    if head < len(visited):
        path = [visited[head]]
        path_action = [actions[head]]
        while parents[head]>0:
            head = parents[head]
            path.insert(0, visited[head])
            path_action.insert(0, actions[head])
        path.insert(0, visited[0])
    return path, path_action

def get_how_much_options_left(visited):
    np_visited = np.array(visited)
    return len(visited)-sum(np_visited[:,1])

def get_state_score(h_scores, new_state):
    name = convert_topology_to_str(new_state.points)
    if name in h_scores.keys():
        return h_scores[name]
    else:
        return 0

def get_index_from_id(id, list_of_states):
    for index, item in enumerate(list_of_states):
        if item[3] == id:
            return index

def bfs_new(start, goal, max_depth=None, with_h=True, h_path="metrics/h_values.txt", one_step=False):
    # using max_depth to limit search. mainly used for data generation.
    #visited[topology_state, score, depth, id, parent_id, action]
    goal_found = False
    if with_h:
        h_scores = load_pickle(h_path)
    score = get_state_score(h_scores, start)
    id = 0
    visited_not_used, parents, actions = [[start, score, 1, id, None, None]], [0], [{}]
    id +=1
    visited_used = []
    if start == goal:
        return [start], []
    head = 0
    while len(visited_not_used) > 0:
        visited_not_used = sorted(visited_not_used, key=itemgetter(1), reverse=True)
        item = visited_not_used[0]
        state = item[0]
        #move the object from not used to used
        visited_used.append(visited_not_used.pop(0))
        if max_depth is not None and item[2] > max_depth:
            continue
        for new_state, action in generate_next(state):
            if one_step and len(new_state.points) > len(state.points) + 2:
                continue
            else:
                append = True
                for visited_state in visited_used:
                    if new_state == visited_state[0]:
                        append = False
                        break
                if not append:
                    for visited_state in visited_not_used:
                        if new_state == visited_state[0]:
                            append = False
                            break
                if append:
                    score = get_state_score(h_scores, new_state)
                    visited_not_used.append([new_state, score, item[2]+1, id, item[3], action])
                    id += 1
                if new_state == goal:
                    break
        if visited_not_used[-1][0] == goal:
            visited_used.append(visited_not_used.pop(-1))
            goal_found = True
            head = len(visited_used)-1
            break
    # backtrack to find solution
    if goal_found:
        id = visited_used[-1][3]
        path = [visited_used[-1][0]]
        path_action = [visited_used[-1][5]]
        index = get_index_from_id(id, visited_used)
        while visited_used[index][4] is not None:
            id = visited_used[index][4]
            index = get_index_from_id(id, visited_used)
            path.insert(0, visited_used[index][0])
            path_action.insert(0, visited_used[index][5])
        #path.insert(0, visited_used[0][0])
        return path, path_action
    else:
        return [],[]

def bfs_all_path(start, goal, max_depth):
    # find all paths connecting start to goal.
    # limit search using max_depth.
    visited, parents, actions = [start], [0], [{}]
    depth_index = [0]
    if start == goal:
        return [([start], [])]
    head = 0
    goal_index = []
    while head < len(visited):
        state = visited[head]
        if head>depth_index[-1]:
            depth_index.append(len(visited)-1)
        if max_depth is not None and len(depth_index) > max_depth:
            break
        for new_state, action in generate_next(state):
            visited.append(new_state)
            parents.append(head)
            actions.append(action)
            if new_state == goal:
                goal_index.append(len(visited)-1)
        head += 1
    # backtrack to find solutions
    paths = []
    for idx in goal_index:
        current = idx
        path = [visited[current]]
        path_action = [actions[current]]
        while parents[current]>0:
            current = parents[current]
            path.insert(0, visited[current])
            path_action.insert(0, actions[current])
        path.insert(0, visited[0])
        paths.append((path, path_action))
    return paths

def bfs_all_path_new(start, goal, max_depth, h_path, with_h=True):
    # find all paths connecting start to goal.
    # limit search using max_depth.
    if with_h:
        h_scores = load_pickle(h_path)
    visited, parents, actions = [start], [0], [{}]
    depth_index = [0]
    if start == goal:
        return [([start], [])]
    head = 0
    goal_index = []
    while head < len(visited):
        state = visited[head]
        if int((len(state.points)-2)/2) >= max_depth:
            head += 1
            continue
        if head>depth_index[-1]:
            depth_index.append(len(visited)-1)
        #if max_depth is not None and len(depth_index) > max_depth:
        #    break
        for new_state, action in generate_next(state):
            visited.append(new_state)
            parents.append(head)
            actions.append(action)
            if new_state == goal:
                goal_index.append(len(visited)-1)
        head += 1
    # backtrack to find solutions
    paths = []
    for idx in goal_index:
        current = idx
        path = [visited[current]]
        path_action = [actions[current]]
        while parents[current]>0:
            current = parents[current]
            path.insert(0, visited[current])
            path_action.insert(0, actions[current])
        path.insert(0, visited[0])
        paths.append((path, path_action))
    return paths


if __name__=="__main__":
    start = AbstractState()
    goal = AbstractState()
    goal.Reide1(0, 1, 1)
    goal.cross(1,2)
    goal.cross(4,2)
    start_time = time.time()
    path, path_action = bfs(start, goal)
    print(path)
    end = time.time()
    print(f"Runtime of the program is {end - start_time}")
    start_time = time.time()
    paths = bfs_all_path(start, goal, max_depth=3)
    print(paths)
    end = time.time()
    print(f"Runtime of the program is {end - start_time}")
    temp=1
