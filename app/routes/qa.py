
import time
from fastapi import APIRouter
from app.batch.InteractiveChatAsyncbatch import batch_ask
import logging
from core.utils.MyUtils import MyUtils
from features.chatbot.data.models.ChatBotModel import ChatBotPayloadModel, ChatBotResponseModel


router = APIRouter(
    prefix="/qabot",
    tags=['qa']
)
appProps = MyUtils.load_properties("general")["app"]
logger = logging.getLogger(appProps["logger"])


@router.post("/ask")
async def ask_llm(model: ChatBotPayloadModel) -> ChatBotResponseModel:
    start = time.time()
    cbread = await batch_ask(model) #will return a serialized dict
    #logger.info(f"type of returned object: {cbread.__dict__}")
    cbresponse = ChatBotResponseModel(**cbread.dict())
    end = time.time() - start
    logger.info(f"{cbresponse.question} took: {end} seconds")
    return cbresponse