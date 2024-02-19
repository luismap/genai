import os
import ray

RAY_PORT = os.environ['CDSW_APP_PORT']
DASHBOARD_PORT = os.environ['CDSW_PUBLIC_PORT']

context = ray.init(num_gpus=1, 
         num_cpus=8,
        dashboard_host= "127.0.0.1",
        dashboard_port= int(DASHBOARD_PORT),
        #address=f"127.0.0.1:{RAY_PORT}"
                  )

@ray.remote
def f(x):
    return x * x

futures = [f.remote(i) for i in range(4)]
print(ray.get(futures)) # [0, 1, 4, 9]

ray.cluster_resources()

context.dashboard_url