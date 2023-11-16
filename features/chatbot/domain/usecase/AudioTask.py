
from features.chatbot.data.controller.AudioController import AudioController
from features.chatbot.data.models.AudioDataModel import AudioDataReadModel


class AudioTask:
    def __init__(self
                 ,audio_ctr: AudioController) -> None:
        self._audio_ctr = audio_ctr

    def transcribe(self, audio_file: str, src_language: str = "english") -> AudioDataReadModel:
        adrm = self._audio_ctr.transcribe(audio_file, src_language)
        return adrm
    
    def translate(self, audio_file: str, src_language: str = "english") -> AudioDataReadModel:
        adrm = self._audio_ctr.translate(audio_file, src_language)
        return adrm