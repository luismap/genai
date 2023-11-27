import asyncio
from fastapi import Body, FastAPI
import logging
import logging.config
import yaml
from app.InteractiveChatAsyncbatch import batch_ask, batch_processing_loop, initialize_model
from core.utils.MyUtils import MyUtils
import concurrent
import time

from features.chatbot.data.models.ChatBotModel import ChatBotPayloadModel, ChatBotReadModel, ChatBotResponseModel

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

@app.post("/ask_llm")
async def ask_llm(model: ChatBotPayloadModel) -> ChatBotResponseModel:
    start = time.time()
    cbread = await batch_ask(model) #will return a serialized dict
    #logger.info(f"type of returned object: {cbread.__dict__}")
    cbresponse = ChatBotResponseModel(**cbread)
    end = time.time() - start
    logger.info(f"{cbresponse.question} took: {end} seconds")
    return cbresponse
