import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
from torch import Tensor
import torch.nn as nn
#from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, InMemoryDataset
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing, TopKPooling, SAGEConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.utils import remove_self_loops, add_self_loops
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


import os
import time
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict


torch.backends.cudnn.benchmark = True

class SAGEConv_my(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(SAGEConv_my, self).__init__(aggr='add') #  "Max" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.act = torch.nn.Tanh()
        self.update_lin = torch.nn.Linear(in_channels + out_channels, out_channels, bias=False)
        self.update_act = torch.nn.ReLU()
        
    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j):
        # x_j has shape [E, in_channels]
        x_j = self.lin(x_j)
        x_j = self.act(x_j)
        
        return x_j

    def update(self, aggr_out, x):
        # aggr_out has shape [N, out_channels]
        new_embedding = torch.cat([aggr_out, x], dim=1)
        new_embedding = self.update_lin(new_embedding)
        new_embedding = self.update_act(new_embedding)
        
        return new_embedding

class Net(torch.nn.Module):
    def __init__(self, input_dim, output_size=63, num_of_msgs=21):
        super(Net, self).__init__()
        #https://stackoverflow.com/questions/58097924/how-to-create-variable-names-in-loop-for-layers-in-pytorch-neural-network
        self.input_dim = input_dim
        self.layers = nn.ModuleList()
        for index in range(num_of_msgs):
            if (index == 0):
                self.layers.append(SAGEConv_my(in_channels=input_dim, out_channels=input_dim))#, root_weight=False, bias=False))
            else:
                self.layers.append(SAGEConv_my(in_channels=input_dim, out_channels=input_dim))#, root_weight=False, bias=False))
            #self.layers.append(TopKPooling(128, ratio=1.))
            
        #self.conv1 = SAGEConv_my(input_dim,128, root_weight=False, bias=False)
        #self.pool1 = TopKPooling(128, ratio=1.)
        #self.conv2 = SAGEConv_my(128,128, root_weight=False, bias=False)
        #self.pool2 = TopKPooling(128, ratio=1.)
        #self.conv3 = SAGEConv_my(128,128, root_weight=False, bias=False)
        #self.pool3 = TopKPooling(128, ratio=1.)
        #self.item_embedding = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embed_dim)
        self.lin1 = torch.nn.Linear(input_dim, 256)
        self.lin2 = torch.nn.Linear(256, 128)
        self.lin3 = torch.nn.Linear(128, output_size)
        #self.bn1 = torch.nn.BatchNorm1d(128)
        #self.bn2 = torch.nn.BatchNorm1d(64)
        self.act1 = nn.ReLU()   
  
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        #x = self.item_embedding(x)
        x = x.squeeze(1)        
        """
        x = torch.tanh(self.conv1(x, edge_index))

        x, edge_index, _, batch, _ , _= self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        x = torch.tanh(self.conv2(x, edge_index))
     
        x, edge_index, _, batch, _ , _= self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        x = torch.tanh(self.conv3(x, edge_index))

        x, edge_index, _, batch, _ , _= self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        x = x1 + x2 + x3
        """
        for layer in self.layers[:]:
            x = layer(x, edge_index)
            x = self.act1(x)
        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act1(x)      
        #x = F.dropout(x, p=0.5, training=self.training)

        x = self.lin3(x)

        return x

class Net_simple(torch.nn.Module):
    def __init__(self,input_dim, output_dim):
        super(Net_simple, self).__init__()
        self.conv1 = GCNConv(input_dim, 16)
        self.conv2 = GCNConv(16, output_dim)

    def forward(self,data):
        x, edge_index = data.x, data.edge_index
        x = F.tanh(self.conv1(x, edge_index))
        #x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def tensor_to_list(x, y):
    dataset = []
    for i in range(len(y)):
        dataset.append((x[i],y[i]))
    return(dataset)

def get_folder_name(part_name):
    for directories in os.listdir(os.getcwd()+"/logs/"):
        if part_name in directories:
            return directories
    raise

def convert_dict_to_numpy(sample_dict):
    index = int(sample_dict['action'][0])
    sample = []
    for i in range(21):
        sample += sample_dict["start"][i*3:(i+1)*3]
        if (i == index):
            sample += sample_dict['action'][1:]
        else:
            sample += [0.,0.,0.]
    gt = sample_dict['end']
    return sample, gt

def preprocessing_dataset(path, ratio=0.9):
    with open(path, "rb") as fp:   # Unpickling
        dataset_dict = pickle.load(fp)
    dataset_list = []
    gt_list = []
    for item in dataset_dict:
        sample, gt = convert_dict_to_numpy(item)
        dataset_list.append(sample)
        gt_list.append(gt)
    
    x_train = dataset_list[:int(ratio*len(dataset_list))]
    y_train = gt_list[:int(ratio*len(gt_list))]
    x_test = dataset_list[int(ratio*len(dataset_list)):]
    y_test = gt_list[int(ratio*len(gt_list)):]

    train_dataset = [(x_train[i], y_train[i]) for i in range(len(x_train))]
    test_dataset = [(x_test[i], y_test[i]) for i in range(len(x_test))]

    return train_dataset, test_dataset

def convert_data_to_graph(raw_dataset):
    dataset = []
    forward_start = [i for i in range(20)]
    forward_end = [i for i in range(1,21)]
    reverse_start = [i for i in range(20,0,-1)]
    reverse_end = [i for i in range(19,-1,-1)]
    start = [*forward_start, *reverse_start]
    end = [*forward_end, *reverse_end]
    for item in raw_dataset:
        x = item[0]
        y = item[1]
        x = torch.tensor(x, dtype=torch.float)
        x = x.view(-1,6)
        y = torch.tensor(y, dtype=torch.float)
        y = y.view(1,-1)
        edge_index = torch.tensor([start, end], dtype=torch.long)
        dataset.append(Data(x=x, y=y, edge_index=edge_index))
    return dataset

class mlp(nn.Module):
    def __init__(self, input_size, output_size):
        super(mlp, self).__init__()
        self.layer1 = nn.Linear(input_size, input_size*4)
        self.layer2 = nn.Linear(input_size*4, input_size*8)
        self.layer3 = nn.Linear(input_size*8, input_size*16)
        self.layer4 = nn.Linear(input_size*16, input_size*8)
        self.layer5 = nn.Linear(input_size*8, input_size*4)
        self.layer6 = nn.Linear(input_size*4, output_size)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        x = self.layer1(x)
        x = self.tanh(x)
        x = self.layer2(x)
        x = self.tanh(x)
        x = self.layer3(x)
        x = self.tanh(x)
        x = self.layer4(x)
        x = self.tanh(x)
        x = self.layer5(x)
        x = self.tanh(x)
        x = self.layer6(x)
        return x

def score_function_mse(out , trues):
    answer = (out - trues) * (out - trues)
    sum = answer.sum().item()
    sum = -sum
    return sum

def get_metrics(best_eval_score: float, eval_score: float, train_loss: float):
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

def train(model: nn.Module, train_loader: DataLoader, eval_loader: DataLoader, train_params, device):
    """
    Training procedure. Change each part if needed (optimizer, loss, etc.)
    :param model:
    :param train_loader:
    :param eval_loader:
    :param train_params:
    """
    metrics = {'train_loss': 0, 'train_score': 0, 'total_norm': 0, 'count_norm': 0}
    best_eval_score = None

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params["lr"]["lr_value"])

    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=train_params["lr"]["lr_step_size"],
                                                gamma=train_params["lr"]["lr_gamma"])
    loss_func = nn.MSELoss()
    for epoch in tqdm(range(train_params['num_epochs'])):
        t = time.time()
        metrics = {'train_loss': 0, 'train_score': 0, 'total_norm': 0, 'count_norm': 0}

        for i, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            output = output.view(-1,63)
            y_hat = data.y.to(device)
            loss = loss_func(y_hat, output)
            loss.backward()
            optimizer.step()

            # Calculate metrics
            metrics['total_norm'] += nn.utils.clip_grad_norm_(model.parameters(), train_params['grad_clip'])
            metrics['count_norm'] += 1

            batch_score = score_function_mse(y_hat, output)
            metrics['train_score'] += batch_score

            metrics['train_loss'] += loss.item() * data.num_graphs

        # Learning rate scheduler step
        scheduler.step()

        # Calculate metrics
        metrics['train_loss'] /= len(train_loader.dataset)

        metrics['train_score'] /= len(train_loader.dataset)

        norm = metrics['total_norm'] / metrics['count_norm']

        model.train(False)
        metrics['eval_score'], metrics['eval_loss'] = evaluate(model, eval_loader, epoch, device)
        model.train(True)

        print("epoch = ", epoch, " Accuracy/Train = ", metrics['train_score']," Accuracy/Validation = ", metrics['eval_score'],\
            " Loss/Train = ", metrics['train_loss'], " Loss/Validation = ", metrics['eval_loss'].item())

        if best_eval_score == None or metrics['eval_score'] > best_eval_score:
            best_eval_score = metrics['eval_score']
            torch.save(model.state_dict(), "model.pt")

    return get_metrics(best_eval_score, metrics['eval_score'], metrics['train_loss'])

@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, epoch: int, device):
    """
    Evaluate a model without gradient calculation
    :param model: instance of a model
    :param dataloader: dataloader to evaluate the model on
    :return: tuple of (accuracy, loss) values
    """
    score = 0
    loss = 0
    out = None
    trues = None

    loss_func = nn.MSELoss()
    for i, data in enumerate(dataloader):
        data = data.to(device)
        output = model(data)
        output = output.view(-1,63)
        y_hat = data.y.to(device)
        loss += loss_func(y_hat, output)
        score += score_function_mse(y_hat, output)
        if (i==0):
            out = output
            trues = y_hat
        else:
            out = torch.cat((out, output),0)
            trues = torch.cat((trues , y_hat),0)
    
    
    score /= len(dataloader.dataset)

    return score, loss

def plot_rope(ax, state, gt=False, save=False):
    state = state.view(-1,3)
    state = state.tolist()
    x = [point[0] for point in state]
    y = [point[1] for point in state]
    if gt:
        ax.plot(x, y, 'ro')
    else:
        ax.plot(x, y, 'bo')

def get_start_state_from_x(state):
    state = state.view(-1,6)
    state = state[:,:3]
    return state

def viz_model_output(model, dataloader, save=False, device="cpu"):
    for _ , (x, y) in enumerate(dataloader):
        if isinstance(x,list):
            x = torch.stack(x)
            x = x.permute(1,0).float()

        if isinstance(y,list):
            y = torch.stack(y)
            y = y.permute(1,0).float()

        if torch.cuda.is_available():
            x = x.to(device)
            y = y.to(device)

        y_hat = model(x)
        break
    cnt = 0 
    for index in range(y_hat.shape[0]):
        if(cnt > 200):
            break
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        plot_rope(ax2, y_hat[index], gt=True)
        plot_rope(ax2, y[index], gt=True)
        x_state = get_start_state_from_x(x[index])
        plot_rope(ax1, x_state, gt=False)
        ax1.set_xlim([-1, 1])
        ax1.set_ylim([-1, 1])
        ax2.set_xlim([-1, 1])
        ax2.set_ylim([-1, 1])
        #ax3.set_xlim([-1, 1])
        #ax3.set_ylim([-1, 1])
        ax1.set_title("start")
        ax2.set_title("prediction")
        #ax3.set_title("GT")
        fig.savefig("datasets/output/"+str(index)+"_new.png")
        cnt+=1
        plt.close(fig)  

def main() -> None:
    """
    Run the code following a given configuration
    :param cfg: configuration file retrieved from hydra framework
    """
    seed = 0
    batch_size = 256
    num_workers = 6
    dataset_path = "datasets/full_dataset.txt"
    parallel = True
    input_size = 126
    output_dim = 3
    train_network = True
    input_dim = 6
    num_of_msgs = 21

    lr_parms = {
        "lr_value": 0.001,
        "lr_gamma": 0.1,
        "lr_step_size": 120
    }
    train_params = {
    "num_epochs": 240,
    "grad_clip": 0.3,
    "lr":lr_parms
    }

    # Set seed for results reproduction
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("device = ", device)

    
    raw_train_dataset, raw_test_dataset = preprocessing_dataset(path=dataset_path)

    train_dataset = convert_data_to_graph(raw_train_dataset)
    test_dataset = convert_data_to_graph(raw_test_dataset)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
    eval_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=num_workers)
    
    # Init model
    model = Net(input_dim, output_dim, num_of_msgs)

    if not train_network:
        init = torch.load("checkpoints/model_large_network_batch_size_2048.pt")
        update_model = {}
        for i in init:
            new_i = i.replace('module.', '')
            update_model[new_i] = init[i]
        model.load_state_dict(update_model)

    if torch.cuda.is_available():
        model = model.to(device)

    # Add gpus_to_use
    if parallel:
        model = torch.nn.DataParallel(model)

    # Report metrics and hyper parameters to tensorboard
    if train_network:
        metrics = train(model, train_loader, eval_loader, train_params, device)
        print(metrics)
    else:
        viz_model_output(model, eval_loader, save=False, device=device)
    temp=1

if __name__ == '__main__':
    raise #[ms] need to update it
    main()



