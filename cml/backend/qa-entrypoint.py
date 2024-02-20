import os

use_vllm = os.environ["use_vllm"]

if (use_vllm == "true"):
    !python3 cml/backend/ray_start_cluster_python.py
    !export PATH=$PATH:/home/cdsw/.local/bin; uvicorn --app-dir app main-qa:app --port $CDSW_APP_PORT
else:
    !export PATH=$PATH:/home/cdsw/.local/bin; uvicorn --app-dir app main-qa:app --port $CDSW_APP_PORT