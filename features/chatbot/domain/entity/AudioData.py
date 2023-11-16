

from typing import List
from pydantic import BaseModel


class AudioData(BaseModel):
    model: str
    source_audio: str
    text: str
    chunks: List[dict]
