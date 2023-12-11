from datetime import datetime
from features.chatbot.domain.entity.AudioData import AudioData, AudioDataPayload


class AudioDataModel(AudioData):
    date_created: str = str(datetime.now())

class AudioDataReadModel(AudioData):
    task: str
    date_read: str = str(datetime.now())

class AudioDataResponseModel(AudioData):
    pass

class AudioDataPayloadModel(AudioDataPayload):
    pass