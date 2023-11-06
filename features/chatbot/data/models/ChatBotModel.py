
from features.chatbot.domain.entity.ChatBot import ChatBot
from datetime import datetime

class ChatBotModel(ChatBot):
    date_created: datetime = datetime.now()

class ChatBotReadModel(ChatBotModel):
    date_read: datetime = datetime.now()


class ChatBotResponseModel(ChatBotReadModel):
    pass
