import copy
import tqdm
import pickle
import argparse

from utils.topology import representation
from utils.general_utils import convert_topology_to_str

def generate_state_from_high_level_actions(state, actions):
    for item in actions:
        #[MS] need to fix the number of parameters I am sending
        #RT1
        if item[0] == "Reide1":
            state.Reide1(item[1], item[2], item[3])
            continue

        #RT2
        elif item[0] == "Reide2":
            if len(item[0]) == 4:
                state.Reide2(item[1], item[2], item[3], item[4])
            else:
                state.Reide2(item[1], item[2], item[3])
            continue

        #Cross
        elif item[0] == "cross":
            state.cross(item[1], item[2], item[3])
            continue
    return state


def get_all_configurations(cross_len):
    all_passibale_actions = []
    #selecet r1 index
    for r1_index in range(cross_len):
        for r1_left in [1,-1]:
            for r1_sign in [1,-1]:
                all_passibale_actions.append(["r1",r1_index, r1_left, r1_sign])

    #r2
    for r2__over_index in range(cross_len):
        for r2__under_index in range(cross_len):
            r2_over_first = None
            if r2__over_index == r2__under_index:
                r2_over_first = -1
            for r2_left in [1,-1]:
                if r2_over_first is None:
                    all_passibale_actions.append(["r2",r2__over_index, r2__under_index, r2_left])
                else:
                    all_passibale_actions.append(["r2",r2__over_index, r2__under_index, r2_left, r2_over_first])
    #cross
    for c__over_index in range(cross_len):
        for c__under_index in range(cross_len):
            for cross_sign in [1,-1]:
                all_passibale_actions.append(["cross",c__over_index, c__under_index, cross_sign])

    return all_passibale_actions


def generate_all_states_one_action(all_passibale_actions, states, actions):
    
    new_states = {}

    for state_key in states.keys():
        #get topology_state
        state = states[state_key][0]
        for action in tqdm.tqdm(all_passibale_actions):
            temp_actions = copy.deepcopy(states[state_key][1])
            befor_action_1 = copy.deepcopy(state)
            if action[0] == "r1":
                try:
                    befor_action_1.Reide1(action[1], action[2], action[3])
                    temp_actions.append(action)
                except:
                    continue
            elif action[0] == "r2":
                if (len(action) == 4):
                    try:
                        befor_action_1.Reide2(action[1], action[2], action[3])
                        temp_actions.append(action)
                    except:
                        continue
                else:
                    try:
                        befor_action_1.Reide2(action[1], action[2], action[3], action[4])
                        temp_actions.append(action)
                    except:
                        continue
            elif action[0] == "cross":
                try:
                    befor_action_1.cross(action[1], action[2], action[3])
                    temp_actions.append(action)
                except:
                    continue
            else:
                print("befor_action_1_fail")

            topology_str = convert_topology_to_str(befor_action_1)
            if topology_str not in new_states and topology_str not in states:
                new_states[topology_str] = ((copy.deepcopy(befor_action_1), temp_actions))

    for key in new_states:
        if key not in states:
            states[key] = new_states[key]

    return states, actions


def generate_all_states(number_of_crosses):
    initial_state_actions = []
    topology_start = generate_state_from_high_level_actions(representation.AbstractState(), initial_state_actions)
    cross_len = number_of_crosses+1
    all_passibale_actions = get_all_configurations(cross_len)

    states = {}
    states[convert_topology_to_str(topology_start)] = ((copy.deepcopy(topology_start), []))
    actions = []
    output_states = []
    for _ in range(number_of_crosses):
        states, actions = generate_all_states_one_action(all_passibale_actions, states, actions)


    for key in states.keys():
        if len(states[key][0].points) == 2*(number_of_crosses+1):
            output_states.append(states[key])
    return output_states


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data Parameters
    parser.add_argument('--number_of_crosses', type=int, default=1)
    parser.add_argument('--output_path', default="metrics/all_1_crosses_states.txt", type=str)

    args = parser.parse_args()

    assert "txt" in args.output_path

    states = generate_all_states(
        number_of_crosses=args.number_of_crosses,
        )

    #save
    with open(args.output_path, "wb") as fp:
        pickle.dump(states, fp)
    