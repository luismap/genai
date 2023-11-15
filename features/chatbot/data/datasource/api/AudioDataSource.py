
from abc import ABC, abstractmethod

from features.chatbot.data.models.AudioDataModel import AudioDataReadModel


class AudioDataSource(ABC):
    @abstractmethod
    def translate(audio_file, src_language: str) -> AudioDataReadModel:
        pass

    @abstractmethod
    def transcribe(audio_file) -> AudioDataReadModel:
        pass

    @abstractmethod
    def is_available() -> bool:
        pass