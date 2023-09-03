import torch

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

        def find_curve(self, configuration, target_topology_state, physics, playground_physics):
            options = []
            actions=self.generate_action(configuration, target_topology_state, physics, playground_physics)
            if not torch.is_tensor(actions):
                raise "need to change the output"
            actions = actions.cpu()
            for index in range(self.batch_size):
                #ensemble uncertainty 
                ensemble_uncertainty = torch.tensor(-1)

                #check predicition and target
                prediction_uncertainty = -1

                #check topolgy ucnertinty
                topology_uncertainty = -1

                sample = {
                    "action": torch.squeeze(actions[index]),
                    "prediction": None,
                    "ensemble_uncertainty": ensemble_uncertainty.item(),
                    "prediction_uncertainty": prediction_uncertainty,
                    "topology_uncertainty": topology_uncertainty,
                    "target_topology_state": target_topology_state
                }
                options.append(sample)

            #select relevant samples
            selected_samples = self._select_samples(options)

            return selected_samples

        def _select_samples(self, samples):
            selected_samples = samples[:6]

            return selected_samples
