# Creating 2 workers in addition to the Head Node
num_workers=2

ray_head_addr = ""
with open("/tmp/rayheadaddress.txt", "r") as file:
    ray_head_addr = file.readline()

# We need to start the ray process with --block else the command completes and the CML Worker terminates
worker_start_cmd = f"!export PATH=$PATH:/home/cdsw/.local/bin; ray start --block --address={ray_head_addr}"



ray_workers = cdsw.launch_workers(
    n=num_workers, 
    cpu=1, 
    memory=1, 
    code=worker_start_cmd,
)

ray_worker_details = cdsw.await_workers(
    ray_workers, 
    wait_for_completion=False)
