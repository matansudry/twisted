#packeges
import torch
from typing import Dict
from torch import Tensor
from omegaconf import DictConfig

#files
from state2action_flow.s2a_utils.types import Metrics

def compute_score_with_logits(logits: Tensor, labels: Tensor) -> Tensor:
    """
    Calculate multiclass accuracy with logits (one class also works)
    :param logits: tensor with logits from the model
    :param labels: tensor holds all the labels
    :return: score for each sample
    """
    logits = torch.max(logits, 1)[1].data  # argmax

    logits_one_hots = torch.zeros(*labels.size())
    if torch.cuda.is_available():
        logits_one_hots = logits_one_hots.cuda()
    logits_one_hots.scatter_(1, logits.view(-1, 1), 1)

    scores = (logits_one_hots * labels)

    return scores


def get_zeroed_metrics_dict() -> Dict:
    """
    :return: dictionary to store all relevant metrics for training
    """
    return {'train_loss_pos': 0, 'train_loss_action': 0, 'train_score_action': 0, 'train_score_pos': 0, 'total_norm': 0,\
         'count_norm': 0, 'eval_score_pos':0, 'eval_score_action':0, 'eval_score':0} 

class TrainParams:
    """
    This class holds all train parameters.
    Add here variable in case configuration file is modified.
    """
    num_epochs: int
    lr: float
    lr_decay: float
    lr_gamma: float
    lr_step_size: int
    grad_clip: float
    save_model: bool
    thershold: float
    pos_ratio: float
    action_ratio: float
    start_topology_evalutaion: int
    stochastic: bool
    train_flow: str

    def __init__(self, **kwargs):
        """
        :param kwargs: configuration file
        """
        self.num_epochs = kwargs['num_epochs']

        self.lr = kwargs['lr']['lr_value']
        self.lr_decay = kwargs['lr']['lr_decay']
        self.lr_gamma = kwargs['lr']['lr_gamma']
        self.lr_step_size = kwargs['lr']['lr_step_size']

        self.grad_clip = kwargs['grad_clip']
        self.save_model = kwargs['save_model']
        self.thershold = kwargs['thershold']
        self.pos_ratio = kwargs['pos_ratio']
        self.action_ratio = kwargs['action_ratio']
        self.start_topology_evalutaion = kwargs['start_topology_evalutaion']
        self.stochastic = kwargs["stochastic"]
        self.train_flow = kwargs["train_flow"]


def get_train_params(cfg: DictConfig) -> TrainParams:
    """
    Return a TrainParams instance for a given configuration file
    :param cfg: configuration file
    :return:
    """
    return TrainParams(**cfg['train'])


def get_metrics(best_eval_score: float, eval_score: float, train_loss: float) -> Metrics:
    """
    Example of metrics dictionary to be reported to tensorboard. Change it to your metrics
    :param best_eval_score:
    :param eval_score:
    :param train_loss:
    :return:
    """
    return {'Metrics/BestAccuracy': best_eval_score,
            'Metrics/LastAccuracy': eval_score,
            'Metrics/LastLoss': train_loss}