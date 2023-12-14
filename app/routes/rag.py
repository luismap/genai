import asyncio
import logging
import time
from typing import List
from fastapi import APIRouter

from core.utils.MyUtils import MyUtils
from features.chatbot.data.models.ChatRagModel import ChatRagPayloadModel, ChatRagResponseModel
from app.batch.RagChatAsyncbatch import batch_ask, rag_uc, batch_processing_loop

router = APIRouter(
    prefix="/rag",
    tags=['rag']
)
appProps = MyUtils.load_properties("general")["app"]
logger = logging.getLogger(appProps["logger"])

logger.info("initializing processing loop")
loop = asyncio.get_event_loop()

logger.info("Starting ragchatbot processing")
loop.create_task(batch_processing_loop(loop))

@router.post("/ask-rag")
async def ask_llm(model: ChatRagPayloadModel) -> ChatRagResponseModel:
    start = time.time()
    crread = await batch_ask(model)
    cbresponse = ChatRagResponseModel(**crread.dict())
    end = time.time() - start
    logger.info(f"{cbresponse.question} took: {end} seconds")
    return cbresponse

@router.post("/web-url-src-upload")
def web_upload(urls: List[str]):
    return rag_uc.load_from_web(urls)

@router.post("/document-upload")
def document_upload(path: str):
    return rag_uc.load_text_from_local(path=path)

@router.post("/vdb-similarity-search")
def similarity_search(content: str):
    answer = rag_uc._chatbot_ctr._vector_db.similarity_search(content)
    return answer

@router.post("/clean-user-context")
def clean_context(user_id: str):
    return rag_uc.clean_context(user_id)

@router.post("/get-user-context")
def get_context_length(user_id: str) -> int:
    return rag_uc.get_context_length(user_id=user_id)