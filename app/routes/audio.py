
import time
from fastapi import APIRouter
import logging
from core.utils.MyUtils import MyUtils
from features.chatbot.data.models.AudioDataModel import AudioDataPayloadModel, AudioDataReadModel, AudioDataResponseModel
from app.batch.AudioTaskAsyncbatch import batch_ask
router = APIRouter(
    prefix="/audio",
    tags=['audio']
)
appProps = MyUtils.load_properties("general")["app"]
logger = logging.getLogger(appProps["logger"])


@router.post("/translate")
async def translate(audio_payload: AudioDataPayloadModel) -> AudioDataResponseModel:
    start = time.time()
    adread = await batch_ask(audio_payload)
    adrm = AudioDataResponseModel(**adread.dict())
    end = time.time() - start
    logger.info(f"{adrm.task} for file {adrm.source_audio} took: {end} seconds")
    return adrm

@router.post("/transcribe")
async def transcribe(audio_payload: AudioDataPayloadModel) -> AudioDataResponseModel:
    start = time.time()
    adread = await batch_ask(audio_payload)
    adrm = AudioDataResponseModel(**adread.dict())
    end = time.time() - start
    logger.info(f"{adrm.task} for file {adrm.source_audio} took: {end} seconds")
    return adrm
