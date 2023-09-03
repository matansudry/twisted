# Run this code only using python3
from utils.topology.representation import AbstractState, reverse_action
import random, copy
import pdb
import time

def generate_next(state):
    # undo cross action
    new_state = copy.deepcopy(state)
    success = new_state.undo_cross(True)
    if success:
        action = {'move':'undo_cross', 'head':True}
        yield new_state, action
    new_state =copy.deepcopy(state)
    success = new_state.undo_cross(False)
    if success:
        action = {'move':'undo_cross', 'head':False}
        yield new_state, action

    # undo R1 action
    for idx in range(1, state.pts+1):
        new_state = copy.deepcopy(state)
        success = new_state.undo_Reide1(idx)
        if success:
            action = {'move':'undo_R1', 'idx':idx}
            yield new_state, action

    # R2 action
    for over_idx in range(1, state.pts):
        for under_idx in range(1, state.pts):
            new_state = copy.deepcopy(state)
            success = new_state.undo_Reide2(over_idx, under_idx)
            if success:
                action = {'move':'undo_R2', 'over_idx':over_idx, 'under_idx':under_idx}
                yield new_state, action


def bfs(start, goal):
    visited, parents, actions = [start], [0], [{}]
    head = 0
    while head < len(visited):
        state = visited[head]
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


def bfs_all_path(start, goal):
    # find all paths from start to goal
    # note goal should be the simpler state.
    visited, parents, actions = [start], [0], [{}]
    head = 0
    goal_index = []
    while head < len(visited):
        state = visited[head]
        for new_state, action in generate_next(state):
            visited.append(new_state)
            parents.append(head)
            actions.append(action)
            if new_state == goal:
                goal_index.append(len(visited)-1)
        head += 1

    # backtrack to find solution
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
    path, path_action = bfs(goal, start)
    reverse_path = path[::-1]
    reverse_path_action = [reverse_action(path_action[i], path[i], path[i+1])
                           for i in range(len(path_action))]
    reverse_path_action = reverse_path_action[::-1]
    print(reverse_path, reverse_path_action)
    end = time.time()
    print(f"Runtime of the program is {end - start_time}")
    start_time = time.time()
    paths = bfs_all_path(goal, start)
    reverse_paths = []
    for path, path_action in paths:
        reverse_path = path[::-1]
        reverse_path_action = [reverse_action(path_action[i], path[i], path[i+1])
                               for i in range(len(path_action))]
        reverse_path_action = reverse_path_action[::-1]
        print(reverse_path, reverse_path_action)
    end = time.time()
    print(f"Runtime of the program is {end - start_time}")
