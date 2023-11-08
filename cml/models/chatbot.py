from core.llm.models.configs.BitsAndBytes import BitsAndBytesConfig
from features.chatbot.data.controller.ChatBotController import ChatBotController
from features.chatbot.data.datasource.Llama2DataSource import Llama2DataSource
from features.chatbot.domain.usecase.InteractiveChat import InteractiveChat


bnb_config = BitsAndBytesConfig()
llm_source = Llama2DataSource(bnb_config)
cb_ctr = ChatBotController([llm_source])
ic = InteractiveChat(cb_ctr)


def ask_me_something(payload):
    question = payload["question"]
    answer = ic.ask_me_something(question)
    return answer