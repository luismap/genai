import os
import cml as cdsw
import ray



DASHBOARD_PORT = os.environ['CDSW_APP_PORT']

# We need to start the ray process with --block else the command completes and the CML Worker terminates
head_start_cmd = f"!ray start --head --block --include-dashboard=true --dashboard-port={DASHBOARD_PORT}"
ray_head = cdsw.launch_workers(
    n=1,
    cpu=2,
    nvidia_gpu=1,
    memory=4,
    code=head_start_cmd,
)

ray_head_details = cdsw.await_workers(
  ray_head, 
  wait_for_completion=False, 
  timeout_seconds=90
)
