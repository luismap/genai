

import logging
from typing import List
from core.utils.MyUtils import MyUtils
from features.chatbot.data.datasource.api.AudioDataSource import AudioDataSource
from features.chatbot.data.models.AudioDataModel import AudioDataPayloadModel, AudioDataReadModel
from features.chatbot.domain.controller.AudioControllerABC import AudioControllerABC


class AudioController(AudioControllerABC):
    def __init__(self
                 ,audio_ds: List[AudioDataSource]) -> None:
        self._app_props = MyUtils.load_properties("general")["app"]
        self._app_state = self._app_props["env"]
        self._logger = logging.getLogger(self._app_props["logger"])
        self._audio_ds = MyUtils.first(audio_ds, lambda ads: ads.is_available == True)
        
        self._logger.info(f" audio controller initialized - {self._audio_ds._model_id}")
        return None
    
    def transcribe(self, audio_payload_models: List[AudioDataPayloadModel]) -> List[AudioDataReadModel]:
        adrm = self._audio_ds.transcribe(audio_payload_models)
        return adrm
    
    def translate(self, audio_payload_models: List[AudioDataPayloadModel]) -> List[AudioDataReadModel]:
        adrm = self._audio_ds.translate(audio_payload_models)
        return adrm