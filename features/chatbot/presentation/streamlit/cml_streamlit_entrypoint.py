import os
import subprocess

serverport=os.getenv('CDSW_APP_PORT')

sbp = subprocess.Popen(["python3.9","-m", "streamlit", "run", os.path.join(
            'features', 'chatbot', 'presentation','streamlit', 'StreamlitFastApi.py'), "--server.address=127.0.0.1", f"--server.port={serverport}"],
                      start_new_session=True)