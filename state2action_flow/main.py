import sys
sys.path.append(".")

import multiprocessing
mp_context = multiprocessing.get_context('spawn')
from omegaconf import OmegaConf
import wandb
import yaml
import os
import tqdm
import argparse

from utils.general_utils import update_parms, convert_s2s_to_s2a_dataset

def main(args) -> None:
    # Read YAML file
    with open(args.cfg, 'r') as stream:
        cfg = yaml.safe_load(stream)

    #os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg['main']['gpus_to_use'])

    os.system('CUDA_LAUNCH_BLOCKING=1')

    import torch
    from torch.utils.data import DataLoader
    torch.backends.cudnn.benchmark = True
    from datetime import datetime
    import numpy as np
    torch.multiprocessing.set_sharing_strategy('file_system')

    from state2action_flow.s2a_utils import main_utils
    from state2action_flow.s2a_train.train_utils import get_train_params
    from state2action_flow.s2a_utils.train_logger import TrainLogger
    from state2action_flow.s2a_models.autoregressive_stochastic_network import autoregressive_stochastic_s2a_netowrk
    from state2action_flow.s2a_utils.dataset_utils import preprocessing_dataset_qpos, CustomDataset_s2a
    from state2action_flow.s2a_train.s2a_train_flow import train_wrapper
    from metrics.s2a_metric import autoregressive_stochastic_topology_metric

    cfg = update_parms(args,cfg)

    wandb.init(project="state2action"+str(cfg["main"]["num_of_links"])+"_links-my-test-project", entity="")

    print(cfg["train"]["batch_size"])

    cfg["main"]["config_length"] = (cfg["main"]["num_of_links"]-1)*2+7
    #config + hot vectore action + x,y,height + position num_of_links*3
    cfg['train']['input_size'] = cfg["main"]["config_length"] + cfg["main"]["num_of_links"] + 3 
    cfg['train']['output_size'] = cfg["main"]["config_length"]
    if cfg["main"]["return_with_init_position"]:
        cfg['train']['input_size'] += 3*(cfg["main"]["num_of_links"]+1)
    
    cfg['main']['experiment_name_prefix']=\
        cfg['main']['experiment_name_prefix']+"_seed="+str(cfg['main']['seed'])+"_lr_value="+str(cfg['train']['lr']["lr_value"])+"_"

    logger = TrainLogger(
        exp_name_prefix=cfg['main']['experiment_name_prefix'],
        logs_dir=cfg['main']['paths']['logs']
        )
    logger.write(OmegaConf.to_yaml(cfg))

    # Set seed for results reproduction
    main_utils.set_seed(cfg['main']['seed'])
    np.random.seed(cfg['main']['seed'])
    batch_size = cfg['train']['batch_size']

    with_qpos = cfg['main']["with_qpos"]

    start=datetime.now()
    # Load dataset
    if cfg['main']["online_dataloader"]:
        training_data = CustomDataset_s2a(dataset_dir=cfg["main"]["paths"]["online_train"])
        train_loader = DataLoader(
            training_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=cfg["main"]["num_workers"]
            )
    else:
        train_dataset, train_topology_list = preprocessing_dataset_qpos(
            cfg,
            paths=cfg['main']['paths']['train'],
            val_topology=True,
            return_qpos=with_qpos,low_limit=cfg["main"]["train_min_cross"],
            return_with_init_position=cfg["main"]["return_with_init_position"],
            max_limit=cfg["main"]["train_max_cross"],
            num_of_samples=None
            )

        train_dataset, train_topology_list = convert_s2s_to_s2a_dataset(train_dataset, train_topology_list)
        if cfg["main"]["save_dataset"]:
            folder_path = "datasets/21_links/ready_data/s2a/train/"
            folder_files = os.listdir(folder_path)
            for file in folder_files:
                if os.path.exists(folder_path+file):
                    os.remove(folder_path+file)
            from utils.general_utils import save_pickle
            for index in tqdm.tqdm(range(len(train_dataset))):
                output_to_save = [train_dataset[index][0],train_dataset[index][1]]
                save_pickle(folder_path+str(index)+".txt", output_to_save)

        train_loader = DataLoader(train_dataset, batch_size, shuffle=cfg['main']['trains'])
        print("train set size = ", len(train_dataset))

    val_dataset, val_topology_list= preprocessing_dataset_qpos(
        cfg,
        paths=cfg['main']['paths']['validation'],
        val_topology=True,
        return_qpos=with_qpos,
        return_with_init_position=cfg["main"]["return_with_init_position"],
        low_limit=cfg["main"]["test_min_cross"],
        max_limit=cfg["main"]["test_max_cross"],
        num_of_samples=None
        )
    val_dataset, val_topology_list = convert_s2s_to_s2a_dataset(val_dataset, val_topology_list)
    eval_loader = DataLoader(val_dataset, batch_size, shuffle=False,)
    print("val set size = ", len(val_dataset))

    print("data load was done in = ", datetime.now()-start)
    
    # Init model
    cfg['train']['input_size'] = len(val_dataset[0][0])
    cfg['train']['output_size'] = len(val_dataset[0][1])

    output_ranges = {
        "height": np.array([0.0001,0.07]),
        "x": np.array([-0.5,0.5]),
        "y": np.array([-0.5,0.5]),
    }

    model = autoregressive_stochastic_s2a_netowrk(
        input_size=cfg['train']['input_size'],
        output_size=cfg['train']['output_size']+3,
        output_ranges=output_ranges,
        dropout=cfg['train']['dropout']
        )
    
    if cfg['main']['load_netowrk']:
        init = torch.load(cfg["main"]["model_path"])
        model_state = init["model_state"]
        update_model = {}
        for i in model_state:
            new_i = i.replace('module.', '')
            update_model[new_i] = model_state[i]
        model.load_state_dict(update_model)

    # TODO: Add gpus_to_use
    if cfg['main']['parallel']:
        model = torch.nn.DataParallel(model)

    if torch.cuda.is_available():
        model = model.cuda()
    else:
        raise

    logger.write(main_utils.get_model_string(model))

    # Run model
    train_params = get_train_params(cfg)

    # Report metrics and hyper parameters to tensorboard
    if cfg['main']['trains']:
        metrics = train_wrapper(
            model=model,
            train_loader=train_loader,
            eval_loader=eval_loader,
            train_params=train_params,
             logger=logger
             )
    else:
        metrics  = autoregressive_stochastic_topology_metric(
            model, 
            eval_loader, 
            num_of_runs=cfg["main"]["num_workers"]
            )
        print("acc = ", metrics)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data Parameters
    parser.add_argument('--cfg', type=str, default="state2action_flow/config/config.yaml")
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--lr_value', type=float, default=None)
    parser.add_argument('--seed', type=float, default=0)

    args = parser.parse_args()
    main(args)
