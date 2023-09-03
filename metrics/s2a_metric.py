#packages
import multiprocessing
mp_context = multiprocessing.get_context('spawn')

import sys
sys.path.append(".")

import torch
from typing import Optional#, Callable
import numpy as np
from dm_control import mujoco

#files
from utils.general_utils import set_physics_state, execute_action_in_curve_with_mujoco_controller,\
    get_position_from_physics, state2topology, convert_topology_state_to_input_vector
from utils.multiprocessing_abstract import WorkerPool, PoolWorker

def autoregressive_stochastic_topology_metric(model, dataloader, num_of_runs=16):
    model.eval()
    queries = []
    queries_id = 0
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
            
        
        list_action_index, list_height, list_x_pos, list_y_pos = [],[],[],[]
        for _ in range(100):
            _, _, ready_action = model(x, train=False)
            action_index, height, x_pos, y_pos = ready_action
            # = model._output_to_sample(actions_index, params)
            list_action_index.append(action_index)
            list_height.append(height)
            list_x_pos.append(x_pos)
            list_y_pos.append(y_pos)
        for index in range(x.shape[0]):
            temp_x = x[index]
            temp_x = temp_x.cpu().tolist()
            action = []
            temp_action_index, temp_height, temp_x_pos, temp_y_pos =\
                torch.zeros(100,1), torch.zeros(100,1),\
                torch.zeros(100,1), torch.zeros(100,1)
            for list_index in range(100):
                temp_action_index[list_index] = list_action_index[list_index][index].item()
                temp_height[list_index] = list_height[list_index][index].item()
                temp_x_pos[list_index] = list_x_pos[list_index][index].item()
                temp_y_pos[list_index] = list_y_pos[list_index][index].item()
            action = (temp_action_index, temp_height, temp_x_pos, temp_y_pos)
            queries.append((queries_id,(temp_x,action)))
            queries_id+=1

    manager_runner = ParallelS2AManager(workers=num_of_runs)
    results =manager_runner.run_queries(queries)
    results = np.asarray(results)
    sum_results = np.sum(results[:,1])

    print(sum_results/len(dataloader.dataset))
    return (sum_results / len(dataloader.dataset))

def stochastic_topology_metric(model, dataloader, num_of_runs=16):
    model.eval()
    queries = []
    queries_id = 0
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


        actions_index, params = model.get_prediction(x)
        action_index, height, x_pos, y_pos = model._output_to_sample(actions_index, params)
        for index in range(x.shape[0]):
            temp_x = x[index]
            temp_x = temp_x.cpu().tolist()
            action = (action_index[index].item(), height[index].item(),\
                 x_pos[index].item(), y_pos[index].item())
            queries.append((queries_id,(temp_x,action)))
            queries_id+=1

    manager_runner = ParallelS2AManager(workers=num_of_runs)
    results =manager_runner.run_queries(queries)
    results = np.asarray(results)
    sum_results = np.sum(results[:,1])

    print(sum_results/len(dataloader.dataset))
    return (sum_results / len(dataloader.dataset))

class ParallelS2AManager(WorkerPool):
    def __init__(self, workers: Optional[int]):
        super().__init__(workers, self._init_worker, check_messages_from_workers=True)

    def process_worker_requests(self, per_worker_requests):
        if len(per_worker_requests) == 0:
            return
        # open the dictionary
        workers, samples = [], []
        for worker_id in per_worker_requests:
            for sample in per_worker_requests[worker_id]:
                workers.append(worker_id)
                samples.append(sample)
        for sample, worker_id in zip(samples, workers):
            self.worker_specific_request_queues[worker_id].put(sample)

    def _init_worker(
            self,
            worker_index,
            requests_queue:multiprocessing.Queue,
            response_queue:multiprocessing.Queue,
            worker_specific_request_queue:multiprocessing.Queue,
            worker_specific_response_queue: multiprocessing.Queue
    ):
        return ParallelS2ARunner(
            worker_index,
            requests_queue,
            response_queue,
            worker_specific_request_queue,
            worker_specific_response_queue
        )

    def _get_stats_message(self):
        pass

    def _reset_stats(self):
        self.batches = []


class ParallelS2ARunner(PoolWorker):
    def __init__(
            self,
            worker_index,
            requests_queue: multiprocessing.Queue,
            response_queue: multiprocessing.Queue,
            worker_specific_request_queue: multiprocessing.Queue,
            worker_specific_response_queue: multiprocessing.Queue
    ):
        super().__init__(worker_index, requests_queue, response_queue, worker_specific_request_queue,
                         worker_specific_response_queue)

    def before_while_loop_init(self):
        env_path = "assets/rope_v3_21_links.xml"
        self.physics = mujoco.Physics.from_xml_path(env_path)

    def do_work(self, query_with_id):
        self.before_while_loop_init()
        query_id, query = query_with_id
        result = self._solve(query)
        return None, (query_id, result)

    def _solve(self, query):
        x = query[0]
        action = query[1]
        sum_true = check_action(action, x, self.physics)
        return sum_true

def check_action(action, x, physics):
    for index in range(100):
        empty_config = np.zeros(93)
        empty_config[:47] = x[:47]
        set_physics_state(physics, empty_config)
        #check if its stochastic or auto
        action_to_check = (action[0][index].item(),action[1][index].item(),\
            action[2][index].item(),action[3][index].item())
        execute_action_in_curve_with_mujoco_controller(physics, action_to_check, num_of_links=21,\
                                                        get_video=False, show_image=False,\
                                                        save_video=False, return_render=False, sample_rate=20,\
                                                        output_path = "outputs/videos/15_links/")
        end_pos = get_position_from_physics(physics)
        topology_end = state2topology(torch.tensor(end_pos))
        if (len(topology_end) > 8):
            return 0
        output = convert_topology_state_to_input_vector(topology_end)
        temp_y = x[113:]
        if np.array_equal(np.array(temp_y), np.array(output)):
            return 1
    return 0