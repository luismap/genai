import os
import cml.workers_v1 as cdsw

DASHBOARD_PORT = os.environ['CDSW_APP_PORT']

# We need to start the ray process with --block else the command completes and the CML Worker terminates
head_start_cmd = f"!export PATH=$PATH:/home/cdsw/.local/bin; ray start --head --block --include-dashboard=true --dashboard-port={DASHBOARD_PORT}"
ray_head = cdsw.launch_workers(
    n=1,
    cpu=1,
    memory=1,
    code=head_start_cmd,
)

ray_head_details = cdsw.await_workers(
  ray_head, 
  wait_for_completion=False, 
  timeout_seconds=90
)

ray_head_ip = ray_head_details['workers'][0]['ip_address']
ray_head_addr = ray_head_ip + ':6379'
ray_head_addr

with open("/tmp/rayheadaddress.txt", "w") as file:
    file.write(ray_head_addr)
    

