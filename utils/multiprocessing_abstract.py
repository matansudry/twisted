import gc
import multiprocessing
import os
import queue
import random
import time
from typing import Optional, Callable

import numpy as np
import psutil


# for now because we want to access handles from the main process (e.g. factory functions) we don't want to use spawn.
# this means that the WorkerPool needs to be initialized before objects that start CUDA.
mp_context = multiprocessing.get_context('spawn')


class PoolWorker(mp_context.Process):
    def __init__(
            self, worker_index, requests_queue: multiprocessing.Queue, response_queue: multiprocessing.Queue,
            worker_specific_request_queue: multiprocessing.Queue, worker_specific_response_queue: multiprocessing.Queue,
    ):
        super().__init__()
        self.worker_index = worker_index

        self.requests_queue = requests_queue
        self.response_queue = response_queue
        self.worker_specific_request_queue = worker_specific_request_queue
        self.worker_specific_response_queue = worker_specific_response_queue

    def before_while_loop_init(self):
        pass

    def do_work(self, query_data):
        return None

    def run(self) -> None:
        # set the seeds
        seed = int(hash(os.getpid()) + hash(time.time()) % (2 ** 32 - 1))
        np.random.seed(seed)
        random.seed(seed)

        self.before_while_loop_init()

        while True:
            self.collect_gc()
            try:
                query_data = self.requests_queue.get(block=True, timeout=0.001)
                response = self.do_work(query_data)
                self.response_queue.put(response)
            except queue.Empty:
                pass
            try:
                termination_message = self.worker_specific_request_queue.get(block=True, timeout=0.001)
                self.worker_specific_response_queue.put(self.worker_index)
                break
            except queue.Empty:
                time.sleep(0.001)
        self.close()

    def collect_gc(self, gc_threshold=75):
        if psutil.virtual_memory().percent >= gc_threshold:
            # print(f'gc called {psutil.virtual_memory().percent}')
            gc.collect()


class WorkerPool:
    def __init__(
            self, workers: Optional[int],
            init_worker_type: Callable[[int, multiprocessing.Queue, multiprocessing.Queue, multiprocessing.Queue, multiprocessing.Queue], PoolWorker],
            check_messages_from_workers: bool = False
    ):
        max_workers = max(os.cpu_count() - 2, 1)
        if workers is None or workers < 1:
            workers = max_workers
        else:
            workers = min(max_workers, workers)

        self.number_of_workers = workers

        self.requests_queue = mp_context.Queue()
        self.response_queue = mp_context.Queue()

        self.worker_specific_request_queues = [mp_context.Queue() for _ in range(workers)]
        self.worker_specific_response_queues = [mp_context.Queue() for _ in range(workers)]
        self.workers = [
            init_worker_type(
                worker_id, self.requests_queue, self.response_queue,
                self.worker_specific_request_queues[worker_id], self.worker_specific_response_queues[worker_id],
            )
            for worker_id in range(workers)
        ]

        self.check_messages_from_workers = check_messages_from_workers

        for w in self.workers:
            w.start()

        self.stats_freq = 1000

        self.total_time = 0.
        self.first_message_wait_time = 0.
        self.followup_message_wait_time = 0.

    def run_queries(self, queries_with_ids):
        assert self.response_queue.empty()

        for query_with_id in queries_with_ids:
            self.requests_queue.put(query_with_id)

        self._reset_stats()
        results = []
        per_worker_requests = {}
        sleep_time = 0.1
        start_time = time.time()
        loop_counter = 0
        while len(results) < len(queries_with_ids):
            messages = self._get_from_response_queue()
            for m in messages:
                if self.check_messages_from_workers:
                    # messages are [worker id (or None), message content]
                    worker_id, message = m
                    if worker_id is None:
                        # if worker id is None, message is the result itself
                        results.append(message)
                        # print(f'len results {len(results)}')
                    else:
                        # process the message and send it to the worker
                        if worker_id not in per_worker_requests:
                            per_worker_requests[worker_id] = [message]
                        else:
                            per_worker_requests[worker_id].append(message)
                else:
                    results.append(m)
            if self.check_messages_from_workers:
                self.process_worker_requests(per_worker_requests)
                per_worker_requests.clear()
            if loop_counter == self.stats_freq:
                loop_counter = 0
                self.total_time = time.time() - start_time
                #print(self._get_stats_message())
                print(f'obtained {len(results)} of {len(queries_with_ids)} ({100 * len(results) / len(queries_with_ids)} %)')
            loop_counter += 1

        self.total_time = time.time() - start_time
        #print(self._get_stats_message())
        return results

    def _get_from_response_queue(self):
        start_sleep_time = time.time()
        messages = [self.response_queue.get(block=True)]
        self.first_message_wait_time += time.time() - start_sleep_time
        while True:
            start_sleep_time = time.time()
            try:
                message = self.response_queue.get(block=True, timeout=0.001)
                messages.append(message)
            except queue.Empty:
                break
            self.followup_message_wait_time += time.time() - start_sleep_time
        return messages

    def _reset_stats(self):
        self.total_time = 0.
        self.first_message_wait_time = 0.
        self.followup_message_wait_time = 0.

    def _get_stats_message(self):
        pass
        #first_relative = 100 * self.first_message_wait_time / self.total_time
        #followup_relative = 100 * self.followup_message_wait_time / self.total_time
        #return f'first message wait time {self.first_message_wait_time} of {self.total_time} ({first_relative} %){os.linesep}followup message wait time {self.followup_message_wait_time} of {self.total_time} ({followup_relative} %)'

    def process_worker_requests(self, per_worker_requests):
        pass

    def close(self):
        assert self.response_queue.empty()

        for worker_specific_request_queue in self.worker_specific_request_queues:
            assert worker_specific_request_queue.empty()
            worker_specific_request_queue.put(None)  # in this queue, any message is termination

        for worker_specific_response_queue in self.worker_specific_response_queues:
            worker_id = worker_specific_response_queue.get(block=True)

        for worker in self.workers:
            worker.join()

        # wait 10 seconds after closing all workers
        time.sleep(10.)