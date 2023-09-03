import yaml
from easydict import EasyDict as edict

def load_config(path: str) -> dict:
    """ load a config file """
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return edict(config)
