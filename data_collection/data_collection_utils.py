import pickle

from utils.general_utils import physics_reset, get_file_name

def convert_topology_to_number_of_crosses(topology):
    crosses = int((len(topology)-2)/2)
    return crosses

def reset_physics_to_init_state(physics):
    physics.reset()
    physics_reset(physics)

def save_trejctory(trejctory, main_folder, num_of_crosses, prefix=""):
    name = get_file_name(num_of_crosses,dataset_path=main_folder, prefix=prefix)
    with open(name, "wb") as fp:
        pickle.dump(trejctory, fp)

def save_sample(sample, main_folder, num_of_crosses, prefix=""):
    name = get_file_name(num_of_crosses,dataset_path=main_folder, prefix=prefix)
    with open(name, "wb") as fp:
        pickle.dump(sample, fp)
