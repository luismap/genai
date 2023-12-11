
from abc import ABC, abstractmethod
from typing import List

from features.chatbot.data.models.AudioDataModel import AudioDataPayloadModel, AudioDataReadModel


class AudioDataSource(ABC):
    @abstractmethod
    def translate(audio_payloads: List[AudioDataPayloadModel] ) -> List[AudioDataReadModel]:
        pass

    @abstractmethod
    def transcribe(audio_payloads: List[AudioDataPayloadModel]) -> AudioDataReadModel:
        pass

    @abstractmethod
    def is_available() -> bool:
        pass