
from features.chatbot.domain.entity.ChatBot import ChatBot
from datetime import datetime

class ChatBotModel(ChatBot):
    date_created: datetime 

class ChatBotReadModel(ChatBotModel):
    date_read: datetime


class ChatBotResponseModel(ChatBotReadModel):
    pass
