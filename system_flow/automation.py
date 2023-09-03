import sys
sys.path.append(".")

from system_flow.high_level_class.high_level import HighLevelPlanner
from system_flow.high_level_class.random_high_level import RandomHighLevelPlanner
from system_flow.system_utils.config_methods import load_config
from utils.general_utils import *
import torch
torch.backends.cudnn.benchmark = True
from metrics.evalution import select_states
import pytorch_lightning
import time
import yaml

HIGH_LEVEL_CATALOG = {
    "random": RandomHighLevelPlanner,
    "guided_by_high_level": HighLevelPlanner
}

def get_problem_idex(states, requried_state):
    for index, item in enumerate(states):
        str_topology = convert_topology_to_str(item[0].points)
        if str_topology==requried_state:
            return index

def print_times_func(system):
    print("time_get_all_high_level_plan =", system.time_get_all_high_level_plan)
    print("time_get_all_initial_topology_states =", system.time_get_all_initial_topology_states)
    print("time_bandit_select_topology_state =", system.time_bandit_select_topology_state)
    print("time_bandit_select_trejctory_from_topology_state =",\
        system.time_bandit_select_trejctory_from_topology_state)
    print("time_follow_plan =", system.time_follow_plan)
    print("time_expand =", system.time_expand)

def update_config_init(cfg):
    cfg['LOW_LEVEL']["STATE2STATE_PARMS"]["config_length"] = (cfg['LOW_LEVEL']["STATE2STATE_PARMS"]["num_of_links"]-1)*2+7
    #config + hot vectore action + x,y,height + position num_of_links*3
    cfg['LOW_LEVEL']['STATE2STATE_PARMS']['input_size'] = cfg['LOW_LEVEL']["STATE2STATE_PARMS"]["config_length"] +\
         cfg['LOW_LEVEL']["STATE2STATE_PARMS"]["num_of_links"] + 3 
    cfg['LOW_LEVEL']['STATE2STATE_PARMS']['output_size'] = cfg['LOW_LEVEL']["STATE2STATE_PARMS"]["config_length"]
    if cfg['LOW_LEVEL']["STATE2STATE_PARMS"]["return_with_init_position"]:
        cfg['LOW_LEVEL']['STATE2STATE_PARMS']['input_size'] += 3*(cfg['LOW_LEVEL']["STATE2STATE_PARMS"]["num_of_links"]+1)
    return cfg

def main():
    if not torch.cuda.is_available():
        raise

    args = argparse_create()

    args.cfg = "system_flow/config/without_uncertainty_with_higher_crosses.yml"
    #"system_flow/config/random_only.yml"
    #"system_flow/config/inverse_only.yml"
    #"system_flow/config/without_uncertainty_with_higher_crosses.yml"
    #"system_flow/config/with_uncertainty_with_higher_crosses.yml" 
    
    cfg = load_config(path=args.cfg)

    cfg["GENERAL_PARAMS"]["config"] = args.cfg
    
    args.env_path = cfg['HIGH_LEVEL'].env_path

    cfg = update_config_init(cfg)

    states = select_states(all_states_path=cfg["GENERAL_PARAMS"]["states_file_path"],\
                    k=cfg["GENERAL_PARAMS"]["k"], h_values_path=cfg["GENERAL_PARAMS"]["h_value_for_states_selection"],\
                    use_unseen=cfg["GENERAL_PARAMS"]["unseen_states"])

    if cfg["GENERAL_PARAMS"]["use_states"][0] != 0 or cfg["GENERAL_PARAMS"]["use_states"][1] != -1:
        states = states[cfg["GENERAL_PARAMS"]["use_states"][0]:cfg["GENERAL_PARAMS"]["use_states"][1]]
        
    print("use states =", cfg["GENERAL_PARAMS"]["use_states"])
    print("env =", cfg['HIGH_LEVEL']["env_path"])
    print("k =", cfg["GENERAL_PARAMS"]["k"])
    print("unseen only =", cfg["GENERAL_PARAMS"]["unseen_states"])

    print_times = False

    print(yaml.dump(edict2dict(cfg), indent=4, default_flow_style=False, sort_keys=True))
    number_of_seeds = 1
    answer_matrix = np.zeros((number_of_seeds, len(states)))
    cnt_matrix = np.zeros((number_of_seeds, len(states)))
    time_matrix = np.zeros((number_of_seeds, len(states)))

    running_time = 1800

    print("cnt_matrix =")
    print(cnt_matrix)
    print("running time is =", running_time)
    if states is not None:
        print("run multiprocerss")
        for new_seed in range(number_of_seeds):
            print("current seed =", new_seed)
            for state_index, state in enumerate(states):
                pytorch_lightning.utilities.seed.seed_everything(seed=new_seed)
                st = time.time()
                system = HIGH_LEVEL_CATALOG[cfg["HIGH_LEVEL"]["type"]](args,cfg)
                system.set_new_goal(state[0])
                if len(cfg['LOW_LEVEL']['STATE2STATE_PARMS'].paths) > 1:
                    answer = run_with_limited_time_new(system.run, (), {}, running_time)
                else:
                    answer = run_with_limited_time(system.run, (), {}, running_time)
                et = time.time()
                answer_matrix[new_seed][state_index] += 1 if answer else 0
                cnt_matrix[new_seed][state_index] += 1
                time_matrix[new_seed][state_index] += et - st
                #system.run()
                
                if print_times:
                    print_times_func(system)
                
                print("answer_matrix =")
                print(answer_matrix)
                print("cnt_matrix =")
                print(cnt_matrix)
                print("time_matrix =")
                print(time_matrix)

                #empty cuda
                try:
                    del system.low_planner
                    del system
                    import gc
                    with torch.no_grad():
                        torch.cuda.empty_cache()
                    gc.collect()
                except:
                    continue

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
    #os.environ["CUDA_VISIBLE_DEVICES"]= "1" #str(args.gpus_to_use)