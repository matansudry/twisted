#packeges
import torch.nn as nn
from torch.utils.data import DataLoader

#files
from state2action_flow.s2a_train.train_utils import TrainParams
from state2action_flow.s2a_utils.types import Metrics
from state2action_flow.s2a_utils.train_logger import TrainLogger
from state2action_flow.s2a_train.autoregressive_stochastic_train import autoregressive_stochastic_train


def train_wrapper(model: nn.Module, train_loader: DataLoader, eval_loader: DataLoader,train_params: TrainParams,\
      logger: TrainLogger) -> Metrics:
    metrics = autoregressive_stochastic_train(model, train_loader, eval_loader, train_params, logger)
    return metrics
