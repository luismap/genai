import cmlextensions.ray_cluster as rc



num_workers=1
head_cpu=2
worker_cpu=2
head_memory=16
worker_memory=16
worker_nvidia_gpu=1
head_nvidia_gpu=1
timeout=900


cluster = rc.RayCluster(num_workers=num_workers, 
                        worker_cpu=worker_cpu, 
                        head_cpu=head_cpu, 
                        worker_memory=worker_memory,
                        head_memory=head_memory,
                        worker_nvidia_gpu=worker_nvidia_gpu,
                        head_nvidia_gpu=head_nvidia_gpu)
cluster.init(timeout)