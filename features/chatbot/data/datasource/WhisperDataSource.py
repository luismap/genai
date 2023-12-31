from itertools import groupby
from typing import List
from core.llm.models.audio.whisper.Whisper import Whisper
from features.chatbot.data.datasource.api.AudioDataSource import AudioDataSource
from features.chatbot.data.models.AudioDataModel import AudioDataPayloadModel, AudioDataReadModel
from transformers import Pipeline

class WhisperDataSource(AudioDataSource):
    def __init__(self
                ,model_id: str = "openai/whisper-large-v3"
                ,task = "automatic-speech-recognition") -> None:
        self._model_id = model_id
        self._whisper_model = Whisper(model_id, task)
        self._audio_model = self._whisper_model.model()
        self._audio_pipeline: Pipeline = self._whisper_model.pipeline_from_pretrained_model(self._audio_model)
        return None
    
    def is_available() -> bool:
        return True
    
    def _group_by_language(self, audio_payloads: List[AudioDataPayloadModel]
                           ) -> List[List[AudioDataPayloadModel]]:
        keyfunc = lambda audio_payload_model: audio_payload_model.language
        grouped = []
        for _, groups in groupby(audio_payloads, keyfunc):
            grouped.append(list(groups))

        return grouped
        
    def transcribe(self, audio_payloads: List[AudioDataPayloadModel] ) -> List[AudioDataReadModel]:
        
        grouped_by_language = self._group_by_language(audio_payloads)
        response = []
        for data in grouped_by_language:
            language = data[0].language
            audio_sources = [apm.source_audio for apm in data]
            transcribed = self._audio_pipeline(audio_sources
                                           ,generate_kwargs={"language": language})
            for apm,transcribed_data in zip(data,transcribed):
                adrm = AudioDataReadModel(
                    model=self._model_id
                    ,source_audio=apm.source_audio
                    ,text=transcribed_data["text"]
                    ,chunks=transcribed_data["chunks"]
                    ,task='transcribe'
                    ,source_language=language
                )
                response.append(adrm)
        return response

    def translate(self,audio_payloads: List[AudioDataPayloadModel]) -> AudioDataReadModel:
        grouped_by_language = self._group_by_language(audio_payloads)
        response = []
        for data in grouped_by_language:
            src_language = data[0].language
            audio_sources = [apm.source_audio for apm in data]

            generate_kwargs={"language": src_language
                         ,"task": "translate"}
            translated = self._audio_pipeline(audio_sources
                                           ,generate_kwargs=generate_kwargs)
            for apm,translated_data in zip(data,translated):
                adrm = AudioDataReadModel(
                    model=self._model_id
                    ,source_audio=apm.source_audio
                    ,text=translated_data["text"]
                    ,chunks=translated_data["chunks"]
                    ,task='translate'
                    ,source_language=src_language
                )
                response.append(adrm)
        return response