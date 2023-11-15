

from core.llm.models.audio.whisper.Whisper import Whisper
from features.chatbot.data.datasource.api.AudioDataSource import AudioDataSource
from features.chatbot.data.models.AudioDataModel import AudioDataReadModel
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
    
    def translate(self, audio_file, src_language: str = "english") -> AudioDataReadModel:
        translation = self._audio_pipeline(audio_file
                                           ,generate_kwargs={"language": src_language})
        adrm = AudioDataReadModel(
            model=self._model_id
            ,source_audio=audio_file
            ,text_decode=translation["text"]
            ,chunks=translation["chunks"]
        )
        return adrm