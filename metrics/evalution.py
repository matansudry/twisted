from operator import itemgetter
from utils.general_utils import load_pickle, convert_topology_to_str

def get_sorted_number_of_visits(number_of_vist_dict:dict) -> list:
    """
    Sort the number of visits

    Args:
        number_of_vist_dict (dict): counter of number of visits

    Returns:
        list: sorted_list
    """
    
    sorted_list = []
    for key in number_of_vist_dict.keys():
        sorted_list.append([key,number_of_vist_dict[key]])
    sorted_list = sorted(sorted_list, key=itemgetter(1), reverse=True)
    return sorted_list

def take_top_k(list_states:list, k:int=2) -> list:
    """
    Take top k states

    Args:
        list_states (list): all states
        k (int, optional): K. Defaults to 2.

    Returns:
        list: top k states
    """
    output = []
    #top k
    for index in range(k):
        output.append(list_states[index])
    #middle k
    number_of_objects = len(list_states)
    for index in range(k):
        output.append(list_states[index+int(number_of_objects/2)])
    #bottom
    for index in range(1,k+1):
        output.append(list_states[-index])
    return output

def find_states_not_visited(top_k_states:list, visited_states:list, path_all_states:str, k:int) -> list:
    """
    find un-visted states

    Args:
        top_k_states (list): top_k_states
        visited_states (list): visited_states
        path_all_states (str): path_all_states
        k (int): K

    Returns:
        list: top_k_states
    """
    cnt = 0
    raw_data = load_pickle(path_all_states)
    for item in raw_data:
        str_topology = convert_topology_to_str(item[0].points)
        if str_topology not in visited_states:
            top_k_states.append([str_topology, 0])
            cnt+=1
            if cnt == k:
                return top_k_states
    print("didnt find enough not visited states")
    raise

def get_eval_states(path_visited_states:str, path_all_states:str, k:int, unseen_only:bool) -> list:
    """
    get states for eval

    Args:
        path_visited_states (str): path_visited_states
        path_all_states (str): path_all_states
        k (int): k
        unseen_only (bool): unseen_only

    Returns:
        list: top_k_states
    """
    raw_data = load_pickle(path_visited_states)
    sorted_states = get_sorted_number_of_visits(raw_data)
    top_k_states = take_top_k(sorted_states, k)
    if unseen_only:
        top_k_states = find_states_not_visited(
            top_k_states=[],
            visited_states=raw_data,
            path_all_states=path_all_states,
            k=k
            )
    return top_k_states

def select_random_states_from_file(file_path:str, k:int)-> list:
    """
    load states file and select K states

    Returns:
        states(list[state]): list of states
    """
    states = []
    raw_states = load_pickle(file_path)
    if k != -1:
        states = raw_states[:k]
    else:
        states = raw_states
    return states

def select_states(all_states_path:str, k:int, use_unseen:bool) -> list:
    """
    select states

    Args:
        all_states_path (str): all_states_path
        k (int): k
        use_unseen (bool): use_unseen

    Returns:
        list: states_after_k
    """
    raw_states = load_pickle(all_states_path)

    if not use_unseen:
        new = []
        for item in raw_states:
            if item[1] != 0:
                new.append(item)
        raw_states = new

    if k != -1:
        states_after_k = take_top_k(raw_states, k=k)
    else:
        states_after_k = raw_states
    
    return states_after_k