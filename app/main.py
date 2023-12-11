import asyncio
from fastapi import Body, FastAPI
import logging
import logging.config
import yaml
import app.batch.InteractiveChatAsyncbatch as ichat_ab
import app.batch.RagChatAsyncbatch as ragchat_ab
import app.batch.AudioTaskAsyncbatch as audio_ab

from core.utils.MyUtils import MyUtils
from app.routes import qa, rag, audio

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

logger.info("initializing processing loop")
loop = asyncio.get_event_loop()

logger.info("Starting chatbot batch processing")
loop.create_task(ichat_ab.batch_processing_loop(loop))

logger.info("Starting ragchatbot processing")
loop.create_task(ragchat_ab.batch_processing_loop(loop))

logger.info("Starting audio processing")
loop.create_task(audio_ab.batch_processing_loop(loop))

logger.info("finished awaiting loop")

@app.get("/")
def read_root():
    if canlog: logger.info("root got call")
    return {"core": "fast api core setup"}

app.include_router(qa.router)
app.include_router(rag.router)
app.include_router(audio.router)
