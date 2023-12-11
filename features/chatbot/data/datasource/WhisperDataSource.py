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
    
    def _group_by_language(audio_payloads: List[AudioDataPayloadModel]
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
            transcribe = self._audio_pipeline(data
                                           ,generate_kwargs={"language": data[0].language})
            for apm,transcribed_data in zip(data,transcribe):
                adrm = AudioDataReadModel(
                    model=self._model_id
                    ,source_audio=apm.audio_path
                    ,text=transcribed_data["text"]
                    ,chunks=transcribed_data["chunks"]
                )
                response.append(adrm)
        return response

    def translate(self, audio_file, src_language: str = "english") -> AudioDataReadModel:
        generate_kwargs={"language": src_language
                         ,"task": "translate"}
        translation = self._audio_pipeline(audio_file
                                           ,generate_kwargs=generate_kwargs)
        adrm = AudioDataReadModel(
            model=self._model_id
            ,source_audio=audio_file
            ,text=translation["text"]
            ,chunks=translation["chunks"]
        )
        return adrm