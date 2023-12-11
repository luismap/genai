

from abc import ABC, abstractmethod
from typing import List

from features.chatbot.data.models.AudioDataModel import AudioDataPayloadModel, AudioDataReadModel


class AudioControllerABC(ABC):
    @abstractmethod
    def transcribe(audio_payload_models: List[AudioDataPayloadModel]) -> List[AudioDataReadModel]:
        pass

    @abstractmethod
    def translate(audio_payload_models: List[AudioDataPayloadModel]) -> List[AudioDataReadModel]:
        pass