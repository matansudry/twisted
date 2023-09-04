#packeges
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
import time
import wandb
import torch

#files
from state2action_flow.s2a_utils.metrics import score_function_mse_binary, score_function_ce
from state2action_flow.s2a_utils.types import Scores
from state2action_flow.s2a_train.train_utils import get_zeroed_metrics_dict, get_metrics, TrainParams
from state2action_flow.s2a_utils.train_logger import TrainLogger
from state2action_flow.s2a_utils.types import Metrics
from metrics.s2a_metric import autoregressive_stochastic_topology_metric

def autoregressive_stochastic_train(model: nn.Module, train_loader: DataLoader, eval_loader: DataLoader,\
    train_params, logger: TrainLogger) -> Metrics:
    """
    Training procedure. Change each part if needed (optimizer, loss, etc.)
    :param model:
    :param train_loader:
    :param eval_loader:
    :param train_params:
    :param logger:
    :return:
    """
    metrics = get_zeroed_metrics_dict()
    best_eval_score = None
    best_train_score = None
    best_topology_score = 0
    score_topology = 0
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params.lr)

    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=train_params.lr_step_size,
                                                gamma=train_params.lr_gamma)
    #loss_func = nn.MSELoss()
    for epoch in tqdm.tqdm(range(train_params.num_epochs)):
        model.train()
        t = time.time()
        metrics = get_zeroed_metrics_dict()


        with torch.autograd.set_detect_anomaly(True):
            for i, (x, y) in enumerate(train_loader):
                if isinstance(x,list):
                    x = torch.stack(x)
                    x = x.permute(1,0).float()

                if isinstance(y,list):
                    y = torch.stack(y)
                    y = y.permute(1,0).float()

                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()

                y_action_hat, y_pos_hat = model(x, train=True, gt=y)
                y_action = y[:,:21]
                y_pos = y[:,21:]
                loss, loss_action, loss_pos, y_action_hat, height_hat, x_hat, y_hat  = model.compute_loss(actions=y_action_hat,\
                    params=y_pos_hat, targets=y, take_mean=True)
                height_hat = height_hat.unsqueeze(dim=1)
                x_hat = x_hat.unsqueeze(dim=1)
                y_hat = y_hat.unsqueeze(dim=1)
                y_pos_hat= torch.cat([height_hat, x_hat, y_hat], dim=1)

                # Optimization step
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_params.grad_clip)
                optimizer.step()

                # Calculate metrics
                metrics['total_norm'] += nn.utils.clip_grad_norm_(model.parameters(), train_params.grad_clip)
                metrics['count_norm'] += 1

                batch_score_pos = score_function_mse_binary(y_pos_hat, y_pos.data, train_params.thershold)
                metrics['train_score_pos'] += batch_score_pos
                batch_score_action = score_function_ce(y_action_hat , y_action.data) #, train_params.thershold)
                metrics['train_score_action'] += batch_score_action.item()

                metrics['train_loss_pos'] += loss_pos.item() *y_pos.shape[1]
                metrics['train_loss_action'] += loss_action.item()

        # Learning rate scheduler step
        scheduler.step()

        # Calculate metrics
        metrics['train_loss_pos'] = metrics['train_loss_pos'] / train_loader.sampler.num_samples
        metrics['train_loss_action'] = metrics['train_loss_action'] / train_loader.sampler.num_samples

        metrics['train_score_action'] = metrics['train_score_action'] / train_loader.sampler.num_samples
        metrics['train_score_pos'] = metrics['train_score_pos'] / train_loader.sampler.num_samples

        norm = metrics['total_norm'] / metrics['count_norm']

        model.train(False)
        metrics['eval_score_pos'], metrics['eval_score_action'], metrics['eval_loss_pos'], metrics['eval_loss_action'] =\
             evaluate_stochastic(model, eval_loader, train_params)
        if epoch%100 == 0 and epoch>train_params.start_topology_evalutaion:
            metrics["eval_score"]  = autoregressive_stochastic_topology_metric(model, eval_loader)
        model.train(True)

        epoch_time = time.time() - t
        logger.write_epoch_statistics(
            epoch=epoch, 
            epoch_time=epoch_time, 
            norm=norm, 
            train_loss_pos=metrics['train_loss_pos'],
            train_loss_action=metrics['train_loss_action'], 
            train_score=metrics['train_score_action'],\
            eval_score=metrics["eval_score"]
            )

        scalars = {'Accuracy/Train/action': metrics['train_score_action'],
                    'Accuracy/Train/pos': metrics['train_score_pos'],
                   'Accuracy/Validation/action': metrics["eval_score"],
                   'Accuracy/Validation/pos': metrics['eval_score_pos'],
                    'Accuracy/Validation': metrics["eval_score"],
                   'Loss/Train pos': metrics['train_loss_pos'],
                   'Loss/Train action': metrics['train_loss_action'],
                   'Loss/Validation pos': metrics['eval_loss_pos'],
                   'Loss/Validation action': metrics['eval_loss_action'],
                   }

        wandb.log({'Train - Accuracy - action': metrics['train_score_action'],
                    'Train - Accuracy - pos': metrics['train_score_pos'],
                   'Train - Loss - pos': metrics['train_loss_pos'],
                   'Train - Loss - action': metrics['train_loss_action'],
                   'Validation - Accuracy - action': metrics['eval_score_action'],
                   'Validation - Accuracy - pos': metrics['eval_score_pos'],
                   'Accuracy/Validation': metrics["eval_score"],
                   'Validation - Loss - pos': metrics['eval_loss_pos'],
                   'Validation - Loss - action': metrics['eval_loss_action']})

        logger.report_scalars(scalars, epoch)

        if best_train_score is None or metrics['train_score_action'] >= best_train_score:
            best_train_score = metrics['train_score_action']
            if train_params.save_model:
                logger.save_model(model, epoch, optimizer, name='model_train_action.pth')

        if best_train_score is None or metrics['train_score_pos'] >= best_train_score:
            best_train_score = metrics['train_score_pos']
            if train_params.save_model:
                logger.save_model(model, epoch, optimizer, name='model_train_pos.pth')

        if best_eval_score is None or metrics['eval_score_action'] >= best_eval_score:
            best_eval_score = metrics['eval_score_action']
            if train_params.save_model:
                logger.save_model(model, epoch, optimizer, name='model_val_action.pth')

        if best_eval_score is None or metrics['eval_score_pos'] >= best_eval_score:
            best_eval_score = metrics['eval_score_pos']
            if train_params.save_model:
                logger.save_model(model, epoch, optimizer, name='model_val_pos.pth')

        if best_topology_score is None or score_topology >= best_topology_score:
            best_topology_score = score_topology
            if train_params.save_model:
                logger.save_model(model, epoch, optimizer, name='model_topology.pth')

    return get_metrics(best_eval_score, metrics['eval_score_action'], metrics['train_loss_pos'])

@torch.no_grad()
def evaluate_stochastic(model: nn.Module, dataloader: DataLoader, train_params: TrainParams) -> Scores:
    """
    Evaluate a model without gradient calculation
    :param model: instance of a model
    :param dataloader: dataloader to evaluate the model on
    :return: tuple of (accuracy, loss) values
    """
    model.eval()
    score_pos = 0
    score_action = 0
    loss_pos = 0
    total_loss_pos = 0
    loss_action=0
    total_loss_action = 0
    for i, (x, y) in enumerate(dataloader):
        if isinstance(x,list):
            x = torch.stack(x)
            x = x.permute(1,0).float()

        if isinstance(y,list):
            y = torch.stack(y)
            y = y.permute(1,0).float()

        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        

        y_action_hat, y_pos_hat = model(x, train=True, gt=y)
        _, loss_action, loss_pos, y_action_hat, height_hat, x_hat, y_hat  = model.compute_loss(actions=y_action_hat,\
            params=y_pos_hat, targets=y, take_mean=True)
        height_hat = height_hat.unsqueeze(dim=1)
        x_hat = x_hat.unsqueeze(dim=1)
        y_hat = y_hat.unsqueeze(dim=1)
        y_pos_hat= torch.cat([height_hat, x_hat, y_hat], dim=1)
        y_action = y[:,:21]
        y_pos = y[:,21:]
        score_pos += score_function_mse_binary(y_pos_hat , y_pos.data, train_params.thershold)
        score_action += score_function_ce(y_action_hat , y_action.data).item()
        total_loss_pos += loss_pos.item() *y_pos.shape[1]
        total_loss_action += loss_action.item() 
    
    score_pos = score_pos / len(dataloader.dataset)
    score_action = score_action / len(dataloader.dataset)
    total_loss_pos = total_loss_pos / len(dataloader.dataset)
    total_loss_action = total_loss_action / len(dataloader.dataset)

    return score_pos, score_action, total_loss_pos, total_loss_action