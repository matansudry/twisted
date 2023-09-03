"""
Schema for config file
"""

CFG_SCHEMA = {
    'main': {
        'experiment_name_prefix': str,
        'seed': int,
        'num_workers': int,
        'parallel': bool,
        'gpus_to_use': int,
        'trains': bool,
        'load_netowrk': bool,
        'output_path': str,
        'paths': {
            'train': list,
            'validation': list,
            'logs': str,
        },
    },
    'train': {
        'num_epochs': int,
        'grad_clip': float,
        'dropout': float,
        'num_hid': int,
        'batch_size': int,
        'save_model': bool,
        'input_size': int,
        'output_size': int,
        'thershold': float,
        'lr': {
            'lr_value': float,
            'lr_decay': int,
            'lr_gamma': float,
            'lr_step_size': int,
            
        },
    },
}