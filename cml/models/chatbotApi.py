from core.llm.models.configs.BitsAndBytes import BitsAndBytesConfig
from features.chatbot.data.datasource.ChatBotLlama2DataSource import Llama2DataSource
from features.chatbot.data.controller.ChatBotController import ChatBotController
from features.chatbot.domain.usecase.InteractiveChat import InteractiveChat


bnb_config = BitsAndBytesConfig()
llm_source = Llama2DataSource(bnb_config)
chat_ctr = ChatBotController([llm_source])
ic = InteractiveChat(chat_ctr)


def ask(payload):
    """
    given a task and a question, produce an answer base on the initialized model
    payload accepted params:
    `task`: can be `predict` or `clean_context`
    clean context will release buffer memory for current chat
    `question`: question to llm
    `history`: if we want history return

    example:
    {
    "task": "predict",
    "question": "give me a list of 5 animals? be short in your answer",
    "history": "false"
    },
    {
    "task": "clean_context",
    "question": "",
    "history": "false"
    }
 
    Args:
        payload (_type_): _description_

    Returns:
        dict: ChatBotReadModel like dict
    """
    if payload["task"] == "predict":
        question = payload["question"]
        history = True if payload["history"] == "true" else False
        answer = ic.ask_me_something(question, history)
        return answer.dict()
    if payload["task"] == "clean_context":
        answer = ic.clean_context()
        return {"cleaned_context": True}
    return None