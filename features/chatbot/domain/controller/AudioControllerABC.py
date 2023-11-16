

from abc import ABC, abstractmethod

from features.chatbot.data.models.AudioDataModel import AudioDataReadModel


class AudioControllerABC(ABC):
    @abstractmethod
    def transcribe(audio_file: str, src_language: str) -> AudioDataReadModel:
        pass

    @abstractmethod
    def translate(audio_file: str, src_language: str) -> AudioDataReadModel:
        pass