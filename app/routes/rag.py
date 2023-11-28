



import logging
import time
from fastapi import APIRouter

from core.utils.MyUtils import MyUtils
from features.chatbot.data.models.ChatRagModel import ChatRagPayloadModel, ChatRagResponseModel
import app.batch.RagChatAsyncbatch  as ragchat_ab

router = APIRouter(
    prefix="/rag",
    tags=['rag']
)
appProps = MyUtils.load_properties("general")["app"]
logger = logging.getLogger(appProps["logger"])



@router.post("/ask-rag")
async def ask_llm(model: ChatRagPayloadModel) -> ChatRagResponseModel:
    start = time.time()
    crread = await ragchat_ab.batch_ask(model)
    cbresponse = ChatRagResponseModel(**crread.dict())
    end = time.time() - start
    logger.info(f"{cbresponse.question} took: {end} seconds")
    return cbresponse
