

from typing import List
from pydantic import BaseModel


class AudioData(BaseModel):
    model: str
    source_audio: str
    text: str
    chunks: List[dict]

class AudioDataPayload(BaseModel):
    audio_path: str
    language: str = 'english'