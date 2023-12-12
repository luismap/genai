
import asyncio
import time
from fastapi import APIRouter
from app.batch.InteractiveChatAsyncbatch import batch_ask, ic, batch_processing_loop
import logging
from core.utils.MyUtils import MyUtils
from features.chatbot.data.models.ChatBotModel import ChatBotPayloadModel, ChatBotResponseModel

router = APIRouter(
    prefix="/qabot",
    tags=['qa']
)
appProps = MyUtils.load_properties("general")["app"]
logger = logging.getLogger(appProps["logger"])


logger.info("initializing processing loop")
loop = asyncio.get_event_loop()

logger.info("Starting chatbot batch processing")
loop.create_task(batch_processing_loop(loop))


@router.post("/ask")
async def ask_llm(model: ChatBotPayloadModel) -> ChatBotResponseModel:
    start = time.time()
    cbread = await batch_ask(model) #will return a serialized dict
    #logger.info(f"type of returned object: {cbread.__dict__}")
    cbresponse = ChatBotResponseModel(**cbread.dict())
    end = time.time() - start
    logger.info(f"{cbresponse.question} took: {end} seconds")
    return cbresponse

@router.post("/clean-user-context")
def clean_context(user_id: str):
    return ic.clean_context(user_id)