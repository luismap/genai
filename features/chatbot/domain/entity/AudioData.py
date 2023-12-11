

from typing import List
from pydantic import BaseModel


class AudioData(BaseModel):
    model: str
    source_audio: str
    text: str
    chunks: List[dict]

class AudioDataPayload(BaseModel):
    task: str = "transcribe" #supporte transcribe, translate
    source_audio: str
    language: str = 'english'