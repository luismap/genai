
from features.chatbot.data.datasource.api.ChatBotDataSource import ChatBotDataSource
from features.chatbot.data.models.ChatBotModel import ChatBotReadModel

class Llama2DataSource(ChatBotDataSource):

    def __init__(self,) -> None:
        super().__init__()

    def generate_answer(question: str) -> ChatBotReadModel:
        
        return super().generate_answer()

