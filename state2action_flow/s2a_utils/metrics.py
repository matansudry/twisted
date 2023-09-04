import torch 
from utils.topology.state_2_topology import state2topology

def acc_score_function_toplogy(out):
    tensor_predictions = torch.stack(out)
    prediction = torch.mean(tensor_predictions,axis = 0)
    return prediction

def score_function_mse(out , trues, change_shape=False):
    if (len(out.size()) == 1):
        out = torch.unsqueeze(out, 0)
    if (len(trues.size()) == 1):
        trues = torch.unsqueeze(trues, 0)
    if change_shape:
        out = out.view(out.shape[0], -1)
        trues = trues.view(trues.shape[0], -1)
    answer = (out - trues) * (out - trues)
    sum = answer.sum().item()
    sum *= -1
    return sum

def score_function_mse_binary(out , trues, treshold):
    
    if (len(out.size()) == 1):
        out = torch.unsqueeze(out, 0)
    if (len(trues.size()) == 1):
        trues = torch.unsqueeze(trues, 0)
    sum_treshold = treshold*trues.shape[1]
    answer = (out - trues) * (out - trues)
    answer = answer.sum(dim=1)
    answer = torch.where(answer < sum_treshold, 1, 0)
    sum = answer.sum().item()
    return sum

def score_function_ce(y_action_hat, y_action):
    y_action_indexs = torch.argmax(y_action[:], dim=1)
    #y_action_hat_index = torch.argmax(y_action_hat[:], dim=1)
    sum = torch.eq(y_action_indexs, y_action_hat).sum()
    return sum

def score_funcation_topology(out , trues, cnt, detailed_eval, conf_mat=False):
    from utils.general_utils import comperae_two_high_level_states
    if conf_mat:
        out = acc_score_function_toplogy(out)
    topology = []
    for i in range(out.shape[0]):
        state = out[i]
        temp_topology = state2topology(state)
        topology.append(temp_topology)#[0].points)
    for index in range(len(trues)):
        trues[index] = [i for i in trues[index] if i]
    if detailed_eval:
        sum = 0
        for index in range(len(topology)):
            temp_topology = trues[index]
            string = ""
            for item_topology in temp_topology:
                string += str(item_topology)
            topology_len = int(len(temp_topology) / 2 -1)
            if topology_len not in cnt.keys():
                cnt[topology_len] = {}
            if string not in cnt[topology_len].keys():
                cnt[topology_len][string] = [0,1]
            else:
                cnt[topology_len][string][1] += 1
            both_topology_are_equal = comperae_two_high_level_states(topology[index], temp_topology)
            sum += both_topology_are_equal
            cnt[topology_len][string][0] += both_topology_are_equal  
    else:
        sum = 0
        for index in range(len(topology)):
            topology_len = len(trues[index])
            if topology_len not in cnt.keys():
                cnt[topology_len] = [0,1]
            else:
                cnt[topology_len][1] += 1
            both_topology_are_equal = comperae_two_high_level_states(topology[index], trues[index])
            sum += both_topology_are_equal
            cnt[topology_len][0] += both_topology_are_equal
    return sum, cnt

def conf_mat_score_funcation_topology(out , trues,init_index):
    from utils.general_utils import comperae_two_high_level_states
    #out = acc_score_function_toplogy(out)
    out = out[0]
    topology = []
    for i in range(out.shape[0]):
        state = out[i]
        temp_topology = state2topology(state)
        topology.append(temp_topology)#[0].points)
    for index in range(len(trues)):
        trues[index] = [i for i in trues[index] if i]
    sum_list = []
    for index in range(len(topology)):
        sum_list.append(comperae_two_high_level_states(topology[index], trues[index+init_index]))
    return sum_list