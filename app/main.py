import asyncio
from fastapi import Body, FastAPI
import logging
import logging.config
import yaml
from app.asyncbatch import batch_processing_loop, batch_translate, initialize_model
from core.utils.MyUtils import MyUtils
import concurrent
import time

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
logger.info("Initial log config in route post!")
loop = asyncio.get_event_loop()
logger.info("initializing processing loop")
loop.create_task(batch_processing_loop(loop))
logger.info("finished awaiting loop")

pool_model = concurrent.futures.ProcessPoolExecutor(max_workers=1, initializer=initialize_model)

@app.get("/")
def read_root():
    if canlog: logger.info("root got call")
    return {"core": "fast api core setup"}

@app.post("/text")
async def translate_text(text: str):
    start = time.time()
    translate = await batch_translate(text, "english")
    end = time.time() - start
    return {"text": translate, "response_time": end}
