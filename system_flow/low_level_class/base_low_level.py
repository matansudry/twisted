import torch
from torch import nn
import numpy as np

from state2state_flow.s2s_models.fc import mlp_equal_size as s2s_network
from utils.general_utils import *
from state2state_flow.s2s_utils.dataset_utils import convert_action_from_index_to_one_hot_vector, fix_yaw_problem
from utils.forward_kinematic import forward_kinematic_from_qpos_torch_batch
from functorch import vmap, combine_state_for_ensemble

class S2SEnsemble(nn.Module):
    def __init__(self, models):
        super(S2SEnsemble, self).__init__()
        self.number_of_models = len(models)
        for model in models:
            model._change_with_foward()
        self.fmodel, self.params, self.buffers = combine_state_for_ensemble(models)
        
    def forward(self, input):
        input.cpu()
        new_input = input.repeat(self.number_of_models,1,1)
        _, x = vmap(self.fmodel)(self.params, self.buffers, new_input)

        batch_size = x.shape[1]
        x = x.view(-1,47)
        output = torch.ones([x.shape[0],22,3], device='cuda:0')
        output *= forward_kinematic_from_qpos_torch_batch(x)
        output = output.view((self.number_of_models, batch_size,-1))
        return output

with torch.no_grad():
    class LowLevelPlanner():
        def __init__(self, cfg, config_length):
            self.cfg = cfg

            #init varibels
            self.paths_to_models = self.cfg["STATE2STATE_PARMS"]["paths"]
            self.input_size = self.cfg["STATE2STATE_PARMS"]["input_size"]
            self.output_size = self.cfg["STATE2STATE_PARMS"]["output_size"]
            self.dropout = self.cfg["STATE2STATE_PARMS"]["dropout"]
            self.num_of_links = self.cfg["STATE2STATE_PARMS"]["num_of_links"]
            self.max_tries = self.cfg["max_tries"]
            self.config_length = config_length
            self.ensemble_prediction = self.cfg["STATE2STATE_PARMS"]["ensemble_prediction"]
            self.batch_size = self.cfg["batch_size"]

            self.init_state2state_nn()
            self.load_state2state_nn()
            if len(self.model_state2state) > 1:
                self.ensemble_network = S2SEnsemble(self.model_state2state)
            else:
                self.ensemble_network = self.model_state2state[0]

        def init_state2state_nn(self):
            self.model_state2state = []
            for _ in range(len(self.paths_to_models)):
                self.model_state2state.append(s2s_network(input_size=self.input_size,\
                    output_size=self.output_size, dropout=self.dropout, num_of_links=self.num_of_links+1))

        def load_state2state_nn(self):
            for index in range(len(self.paths_to_models)):
                init = torch.load(self.paths_to_models[index])
                model_state = init["model_state"]
                update_model = {}
                for i in model_state:
                    new_i = i.replace('module.', '')
                    update_model[new_i] = model_state[i]
                self.model_state2state[index].load_state_dict(update_model)
                self.model_state2state[index].train(False)
                self.model_state2state[index] = self.model_state2state[index].cuda()
                self.model_state2state[index].share_memory()

        def find_curve(self, configuration, target_topology_state, physics, playground_physics):
            options = []
            actions=self.generate_action(configuration, target_topology_state, physics, playground_physics)
            if not torch.is_tensor(actions):
                raise "need to change the output"
            batch = self._create_s2s_input_batch(configuration, actions, playground_physics)
            predictions, ensemble_uncertainties, tensor_predictions = self._get_s2s_prediction_and_uncertainty_batch(batch)
            actions = actions.cpu()
            predictions = predictions.cpu()
            tensor_predictions = tensor_predictions.cpu()
            for index in range(self.batch_size):
                #ensemble uncertainty 
                if self.cfg["Calculate_Uncertainty"]["ensemble"]:
                    ensemble_uncertainty=ensemble_uncertainties[index]
                else:
                    ensemble_uncertainty = torch.tensor(-1)

                #check predicition and target
                if self.cfg["Calculate_Uncertainty"]["prediction"]:
                    prediction_uncertainty = comperae_two_high_level_states(\
                    convert_pos_to_topology(predictions[index]), target_topology_state)
                else:
                    prediction_uncertainty = -1

                #check topolgy ucnertinty
                if self.cfg["Calculate_Uncertainty"]["topology"]:
                    topology_uncertainty = uncertainty_score_function_topology(\
                tensor_predictions[:,index,:], target_topology_state)
                else:
                    topology_uncertainty = -1

                sample = {
                    "action": torch.squeeze(actions[index]),
                    "prediction": predictions[index],
                    "ensemble_uncertainty": ensemble_uncertainty.item(),
                    "prediction_uncertainty": prediction_uncertainty,
                    "topology_uncertainty": topology_uncertainty,
                    "target_topology_state": target_topology_state
                }
                options.append(sample)

            #select relevant samples
            selected_samples = self._select_samples(options)

            return selected_samples

        def _sort_actions(self,samples):

            #order ensemble_uncertainty from low to high
            if self.cfg["Select_Samples"]["Sort_Actions"]["sort_configuration_uncertainty"]: 
                samples = sorted(samples, key=lambda x: x['ensemble_uncertainty'],\
                    reverse=self.cfg["Select_Samples"]["Sort_Actions"]["sort_configuration_uncertainty_reverse"]) 
            
            #order topology_uncertainty from high to low and prediction_uncertainty from True to False
            samples = sorted(samples, key=lambda x: (x['topology_uncertainty'],x['prediction_uncertainty']),\
                 reverse=True)

            #print dis
            dis = []
            for sample in samples:
                dis.append(sample['topology_uncertainty'])
            number_of_crosses = calculate_number_of_crosses_from_topology_state(samples[0]['target_topology_state'])
            print("target_topology_state =", number_of_crosses, ", dis of topology uncertainty =", dis)
            
            return samples

        def _select_samples(self, samples):
            #cnt = len(samples)
            if self.cfg["Select_Samples"]["Sort_Actions"]["enable"]:
                samples = self._sort_actions(samples)
            selected_samples = samples[:6]

            return selected_samples

        def _get_s2s_prediction_and_uncertainty_batch(self, batch):
            uncertainty = []
            tensor_predictions = self.ensemble_network(batch)

            if len(self.model_state2state) == 1:
                tensor_predictions = tensor_predictions[1]
                tensor_predictions = torch.unsqueeze(tensor_predictions, 0)


            #get mean prediction
            if self.ensemble_prediction is None:
                mean_predictions = torch.mean(tensor_predictions, axis=0)
            else:
                mean_predictions = tensor_predictions[self.ensemble_prediction]

            #get uncertainty
            if len(self.paths_to_models) > 1 and self.cfg["Calculate_Uncertainty"]["ensemble"]:
                temp = torch.std(tensor_predictions, axis=0)
                uncertainty = temp.sum(dim=1)
            else:
                uncertainty = None

            return mean_predictions, uncertainty, tensor_predictions

        def _create_s2s_input_batch(self, primitive_state, actions, playground_physics):
            batch = []
            sample = np.array(primitive_state)
            sample = sample[:self.config_length]
            sample = fix_yaw_problem(sample,config_length=self.config_length)
            pos = convert_qpos_to_xyz(playground_physics, primitive_state)
            pos[:,:2] -= sample[:2]
            pos = np.reshape(pos, (-1))
            sample[:2] = 0.0
            for row in range(actions.shape[0]):
                one_hot_action = convert_action_from_index_to_one_hot_vector(actions[row],\
                    num_of_links=self.num_of_links)
                new_sample = np.concatenate((sample, one_hot_action,pos))
                batch.append(new_sample)
            numpy_batch = np.array(batch)
            batch = torch.tensor(numpy_batch, device="cuda").float()
            return batch

        def _get_s2s_uncertainty(self, tensor_predictions):
            temp_uncertainty = torch.std(tensor_predictions,axis = 0)
            uncertainty = temp_uncertainty.sum().item()
            return uncertainty
