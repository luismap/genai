
from pydantic import BaseModel
from features.chatbot.domain.entity.ChatBot import ChatBot, ChatBotPayload
from datetime import datetime

class ChatBotModel(ChatBot):
    date_created: str = str(datetime.now())

class ChatBotReadModel(ChatBot):
    date_read: str = str(datetime.now())


class ChatBotResponseModel(BaseModel):
    answer: str
    model_use: str
    question: str

class ChatBotPayloadModel(ChatBotPayload):
    pass