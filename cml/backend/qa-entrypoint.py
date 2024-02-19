import os
use_vllm = os.environ["use_vllm"]
if (use_vllm):
    !python3 cml/backend/ray_start_cluster_cdsw.py
else:
    !uvicorn --app-dir app main-qa:app --port $CDSW_APP_PORT

