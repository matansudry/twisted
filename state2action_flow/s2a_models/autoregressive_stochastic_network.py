import torch
from torch import nn, Tensor
from torch.distributions import Categorical, Normal


class autoregressive_stochastic_s2a_netowrk(nn.Module):
    def __init__(self, input_size, output_size, output_ranges, minimal_std=0.01, dropout=0.0, device="cuda"):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.output_size = output_size
        self.mixture_count = 4 #mixture_count3
        self.minimal_std = minimal_std
        self.action_loss = nn.CrossEntropyLoss()

        #defining ranges
        self.output_range_height = output_ranges["height"]
        self.output_range_x = output_ranges["x"]
        self.output_range_y = output_ranges["y"]
        self.output_height_min, self.output_height_max = self.output_range_height
        self.output_x_min, self.output_x_max = self.output_range_x
        self.output_y_min, self.output_y_max = self.output_range_y

        #moving ranges to cuda
        self.output_height_min = torch.tensor(self.output_height_min).to(self.device)
        self.output_height_max = torch.tensor(self.output_height_max).to(self.device)
        self.output_x_min = torch.tensor(self.output_x_min).to(self.device)
        self.output_x_max = torch.tensor(self.output_x_max).to(self.device)
        self.output_y_min = torch.tensor(self.output_y_min).to(self.device)
        self.output_y_max = torch.tensor(self.output_y_max).to(self.device)

        #network architacture
        self.input_action_layer = nn.Linear(self.input_size, 512) #index select
        self.input_x_layer = nn.Linear(self.input_size+21+1, 512) # x select
        self.input_y_layer = nn.Linear(self.input_size+21+1+1, 512) # y select
        self.input_z_layer = nn.Linear(self.input_size+21, 512) # z select

        self.action_layer2 = nn.Linear(512, 1024)
        self.action_layer3 = nn.Linear(1024, 2048)
        self.action_layer4 = nn.Linear(2048, 2048)
        self.action_layer5 = nn.Linear(2048, 2048)

        self.z_layer2 = nn.Linear(512, 1024)
        self.z_layer3 = nn.Linear(1024, 2048)
        self.z_layer4 = nn.Linear(2048, 2048)
        self.z_layer5 = nn.Linear(2048, 2048)

        self.x_layer2 = nn.Linear(512, 1024)
        self.x_layer3 = nn.Linear(1024, 2048)
        self.x_layer4 = nn.Linear(2048, 2048)
        self.x_layer5 = nn.Linear(2048, 2048)
    
        self.y_layer2 = nn.Linear(512, 1024)
        self.y_layer3 = nn.Linear(1024, 2048)
        self.y_layer4 = nn.Linear(2048, 2048)
        self.y_layer5 = nn.Linear(2048, 2048)

        self.output_action_layer = nn.Linear(2048, 21) #index select
        self.output_x_layer = nn.Linear(2048, 2) # x select
        self.output_y_layer = nn.Linear(2048, 2) # y select
        self.output_z_layer = nn.Linear(2048, 2) # z select
        self.activation = nn.LeakyReLU(0.2)
        self.dropout_value = dropout
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, train=True, gt=None):
        action_output = self.forward_action(input)
        if train:
            action_gt = gt[:,:21]
            z_input = torch.cat((input,action_gt),1)
        else:
            action_dis = Categorical(action_output)
            action = action_dis.sample()
            action_features = torch.zeros(action.shape[0], 21, device="cuda")
            for row_index in range(action.shape[0]):
                action_features[row_index,action[row_index].item()] = 1
            z_input = torch.cat((input,action_features),1)

        z_output = self.forward_z(z_input)
        if train:
            z_gt = gt[:,22]
            z_gt = z_gt.view(-1,1)
            x_input = torch.cat((input,action_gt,z_gt),1)
        else:
            z_dis = Normal(z_output[:,0], self._make_positive(z_output[:,1]))
            z = z_dis.sample()
            z = z.view(-1,1)
            x_input = torch.cat((input,action_features,z),1)

        x_output = self.forward_x(x_input)
        if train:
            x_gt = gt[:,23]
            x_gt = x_gt.view(-1,1)
            y_input = torch.cat((input, action_gt, z_gt, x_gt),1)
        else:
            x_dis = Normal(x_output[:,0], self._make_positive(x_output[:,1]))
            x = x_dis.sample()
            x = x.view(-1,1)
            y_input = torch.cat((input,action_features, z, x),1)
        
        y_output = self.forward_y(y_input)
        if not train:
            y_dis = Normal(y_output[:,0], self._make_positive(y_output[:,1]))
            y = y_dis.sample()
            y = y.view(-1,1)
        pos_output = torch.cat((z_output, x_output, y_output),1)
        if train:
            return action_output, pos_output
        else:
            action = action.view(-1)
            z = z.view(-1)
            x = x.view(-1)
            y = y.view(-1)
            return action_output, pos_output, (action, z, x, y)

    def forward_action(self, input):
        x = self.input_action_layer(input)
        x = self.activation(x)
        x = self.action_layer2(x)
        x = self.activation(x)
        x = self.action_layer3(x)
        x = self.activation(x)
        x = self.action_layer4(x)
        x = self.activation(x)
        if self.dropout_value > 0:
            x = self.dropout(x) 
        x = self.action_layer5(x)
        x = self.activation(x)
        output = self.output_action_layer(x)
        output = self.softmax(output)
        return output

    def forward_x(self, input):
        x = self.input_x_layer(input)
        x = self.activation(x)
        x = self.x_layer2(x)
        x = self.activation(x)
        x = self.x_layer3(x)
        x = self.activation(x)
        x = self.x_layer4(x)
        x = self.activation(x)
        if self.dropout_value > 0:
            x = self.dropout(x) 
        x = self.x_layer5(x)
        x = self.activation(x)
        output = self.output_x_layer(x)
        return output

    def forward_y(self, input):
        x = self.input_y_layer(input)
        x = self.activation(x)
        x = self.y_layer2(x)
        x = self.activation(x)
        x = self.y_layer3(x)
        x = self.activation(x)
        x = self.y_layer4(x)
        x = self.activation(x)
        if self.dropout_value > 0:
            x = self.dropout(x) 
        x = self.y_layer5(x)
        x = self.activation(x)
        output = self.output_y_layer(x)
        return output

    def forward_z(self, input):
        x = self.input_z_layer(input)
        x = self.activation(x)
        x = self.z_layer2(x)
        x = self.activation(x)
        x = self.z_layer3(x)
        x = self.activation(x)
        x = self.z_layer4(x)
        x = self.activation(x)
        if self.dropout_value > 0:
            x = self.dropout(x) 
        x = self.z_layer5(x)
        x = self.activation(x)
        output = self.output_z_layer(x)
        return output

    def compute_loss(self, actions, params, targets, take_mean=True):
        action_dis, height_dis, x_dis, y_dis = self._output_to_dist(action=actions, params=params)
        if not isinstance(targets, Tensor):
            print("gt is not tensor")
        action_loss = -action_dis.log_prob(torch.argmax(targets[:,:21], dim=1))
        height_loss = -height_dis.log_prob(targets[:,21])
        x_loss = -x_dis.log_prob(targets[:,22])
        y_loss = -y_dis.log_prob(targets[:,23])
        loss = action_loss + height_loss + x_loss + y_loss
        loss_pos = sum(height_loss + x_loss + y_loss)
        loss_action = sum(action_loss)
        action, height, x, y = self._output_to_sample(temp_action=actions, temp_pos=params)
        if take_mean:
            return loss.mean(), loss_action, loss_pos, action, height, x, y
        return loss, action, height, x, y

    def _output_to_sample(self, temp_action, temp_pos, deterministic=False):
        if deterministic:
            action =  torch.argmax(temp_action[:], dim=1)
            pos_prediction = temp_pos.view(-1,3,2)
            height = pos_prediction[:,0,0]
            x = pos_prediction[:,1,0]
            y = pos_prediction[:,2,0]
        else:
            action_dis, height_dis, x_dis, y_dis = self._output_to_dist(temp_action, temp_pos)
            action = action_dis.sample()
            height = height_dis.sample()
            x = x_dis.sample()
            y = y_dis.sample()
        height, x, y = self.clip_sample(height, x, y)
        return action, height, x, y

    def _output_to_dist(self, action, params):
        action_dis = Categorical(action)
        height_params = params[:,:2]
        height_dis = Normal(height_params[:,0], self._make_positive(height_params[:,1]))
        x_params = params[:,2:4]
        x_dis = Normal(x_params[:,0], self._make_positive(x_params[:,1]))
        y_params = params[:,4:6]
        y_dis = Normal(y_params[:,0], self._make_positive(y_params[:,1]))
        return action_dis, height_dis, x_dis, y_dis

    def get_prediction(self, states, train=False):
        actions, params = self(states, train)
        return actions, params    

    def clip_sample(self, height, x, y):
        height = torch.clamp(height, min=self.output_height_min, max=self.output_height_max)
        x = torch.clamp(x, min=self.output_x_min, max=self.output_x_max)
        y = torch.clamp(y, min=self.output_y_min, max=self.output_y_max)
        return height, x, y

    @staticmethod
    def _make_positive(x: Tensor):
        x = torch.exp(x)
        x = x + 1.e-5
        return x