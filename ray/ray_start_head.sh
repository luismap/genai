#!/bin/bash

export PATH=$PATH:/home/cdsw/.local/bin

ray start --head --num-cpus=2 --num-gpus=1 --include-dashboard=true \
--dashboard-port=$CDSW_READONLY_PORT

cat /tmp/ray/ray_current_cluster > cluster_info.txt

echo ""
echo "https://read-only-$CDSW_ENGINE_ID.$CDSW_DOMAIN"

