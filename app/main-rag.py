from fastapi import Body, FastAPI
import logging
import logging.config
import yaml

from core.utils.MyUtils import MyUtils
from app.routes import rag

canlog = True
appProps = MyUtils.load_properties("general")["app"]

#core setup for async

#fast api
app = FastAPI()

with open("logging.yaml", 'rt') as f:
    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)

# Get an instance of the logger and use it to write a log!
# Note: Do this AFTER the config is loaded above or it won't use the config.
logger = logging.getLogger(appProps["logger"])
logger.info("Initial log config in root!")

logger.info("finished awaiting loop")

@app.get("/")
def read_root():
    if canlog: logger.info("root got call")
    return {"core": "fast api core setup"}

app.include_router(rag.router)