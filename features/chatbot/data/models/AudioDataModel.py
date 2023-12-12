from datetime import datetime
from features.chatbot.domain.entity.AudioData import AudioData, AudioDataPayload


class AudioDataModel(AudioData):
    date_created: str = str(datetime.now())

class AudioDataReadModel(AudioData):
    task: str
    source_language: str
    date_read: str = str(datetime.now())

class AudioDataResponseModel(AudioData):
    task: str
    source_language: str
    model: str
    pass

class AudioDataPayloadModel(AudioDataPayload):
    pass