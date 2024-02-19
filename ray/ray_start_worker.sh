

#!/bin/bash

export PATH=$PATH:/home/cdsw/.local/bin

address=`cat cluster_info.txt`
echo $address
ray start --address=$address
