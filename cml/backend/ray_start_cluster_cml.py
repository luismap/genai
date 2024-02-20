import os
import cml.workers_v1 as cdsw

!bash ray_start_head.sh

with open("cluster_info.txt", "r") as file:
  ray_head_addr = file.readline()

worker_start_cmd = f"!export PATH=$PATH:/home/cdsw/.local/bin; ray start --block --address={ray_head_addr}"

num_workers = os.environ[
    "num_workers"
]  # default to 1. 1 ray head and 1 worker. tensors needs a multiple of attentions layers.
memory = os.environ["worker_ram_memory"]  # default to 16
cpu = os.environ["worker_cpu"]  # defaults to 8
gpu = os.environ["worker_gpu"]  # defaults to 1
timeout_seconds = 900  # timeout for workers api to wait for resources assingment


ray_workers = cdsw.launch_workers(
    n=num_workers, 
    cpu=cpu, 
    memory=memory,
    nvidia_gpu = gpu,
    code=worker_start_cmd
)

ray_worker_details = cdsw.await_workers(
    ray_workers, 
    wait_for_completion=False,
    timeout_seconds=timeout_seconds)

ray_worker_details
