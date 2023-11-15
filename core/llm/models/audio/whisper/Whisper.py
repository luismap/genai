from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, PreTrainedModel
from transformers import WhisperProcessor, Pipeline
import torch

class Whisper:
    def __init__(self
                ,model_id: str = "openai/whisper-large-v3"
                ,task = "automatic-speech-recognition") -> None:
        """"
        Interact with openai whisper models hosted on huggingface
        Args:
            model_id (str, optional): model id. Defaults to "openai/whisper-large-v3".
            task (str, optional): supported task. Defaults to "automatic-speech-recognition".
        """
        self._model_id = model_id
        self._model_task = task
        self._processor: WhisperProcessor = AutoProcessor.from_pretrained(self._model_id)
    
    def model(self):
        audio_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self._model_id
            ,torch_dtype=torch.float16
            ,low_cpu_mem_usage=True
            ,use_safetensors=True
            ,device_map="auto"
            )
        return audio_model
    
    def pipeline_from_pretrained_model(self
                                       ,model: PreTrainedModel
                                       ,task: str = "automatic-speech-recognition"
                                       ) -> Pipeline:
        #TODO kwargs all parameters
        pline = pipeline(
                task,
                model=model,
                tokenizer=self._processor.tokenizer,
                feature_extractor=self._processor.feature_extractor,
                max_new_tokens=128,
                chunk_length_s=30,
                batch_size=16,
                return_timestamps=True,
                torch_dtype=torch.float16,
                )
        return pline