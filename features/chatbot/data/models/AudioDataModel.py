from datetime import datetime
from features.chatbot.domain.entity.AudioData import AudioData


class AudioDataModel(AudioData):
    date_created: str = str(datetime.now())

class AudioDataReadModel(AudioData):
    date_read: str = str(datetime.now())

class AudioDataResponseModel(AudioData):
    pass