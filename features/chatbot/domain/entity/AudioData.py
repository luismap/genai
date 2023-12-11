

from typing import List
from pydantic import BaseModel


class AudioData(BaseModel):
    model: str
    source_audio: str
    text: str
    chunks: List[dict]
    task: str

class AudioDataPayload(BaseModel):
    task: str = "transcribe" #supporte transcribe, translate
    audio_path: str
    language: str = 'english'