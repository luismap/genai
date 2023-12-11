
from typing import List
from features.chatbot.data.controller.AudioController import AudioController
from features.chatbot.data.models.AudioDataModel import AudioDataPayloadModel, AudioDataReadModel


class AudioTask:
    def __init__(self
                 ,audio_ctr: AudioController) -> None:
        self._audio_ctr = audio_ctr

    def transcribe(self, audio_payload_models: List[AudioDataPayloadModel]) -> List[AudioDataReadModel]:
        adrm = self._audio_ctr.transcribe(audio_payload_models)
        return adrm
    
    def translate(self, audio_payload_models: List[AudioDataPayloadModel]) -> List[AudioDataReadModel]:
        adrm = self._audio_ctr.translate(audio_payload_models)
        return adrm