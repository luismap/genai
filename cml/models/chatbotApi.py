import asyncio
from app.asyncbatch import batch_ask, batch_processing_loop, clean_user_context
from features.chatbot.data.models.ChatBotModel import ChatBotPayloadModel

loop = asyncio.get_event_loop()
loop.create_task(batch_processing_loop(loop))


async def ask(payload):
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
    "user_id": "default",
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
        model = ChatBotPayloadModel(**payload)
        answer = await batch_ask(model) #will return a serialized dict
        return answer.dict()
    if payload["task"] == "clean_context":
        answer = clean_user_context(payload["user_id"])
        return {"cleaned_context": answer}
    return None