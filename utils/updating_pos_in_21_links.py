import sys
sys.path.append(".")

import tqdm 
import pickle
import os
from utils.general_utils import *
from dm_control import mujoco

def main():
    args = argparse_create()
    physics = mujoco.Physics.from_xml_path(args.env_path_21)
    paths = ["datasets/21_links/old/new_test_with_topology/", "datasets/21_links/old/new_train_with_topology/"]
    new_paths = ["datasets/21_links/new/test/", "datasets/21_links/new/train/"]
    for index, path in enumerate(paths):
        files = os.listdir(path)
        for file in tqdm.tqdm(files):
            with open(path+file, "rb") as fp:   # Unpickling
                try:
                    dataset_dict = pickle.load(fp)
                except:
                    print(file)
                    raise
            if isinstance(dataset_dict, dict):
                for item in dataset_dict.values():
                    with physics.reset_context():
                        physics.set_state(item['start_data'])
                    item['start'] = get_position_from_physics(physics)
                    with physics.reset_context():
                        physics.set_state(item['end_data'])
                    item['end'] = get_position_from_physics(physics)
                    
                with open(new_paths[index]+file, "wb") as fp:
                    pickle.dump(dataset_dict, fp)

if __name__ == '__main__':
    main()




