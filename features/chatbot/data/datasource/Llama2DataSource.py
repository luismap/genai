
from core.llm.models.llama2.Llama2Huggingface import Llama2Hugginface
from features.chatbot.data.datasource.api.ChatBotDataSource import ChatBotDataSource
from features.chatbot.data.models.ChatBotModel import ChatBotReadModel
from core.llm.models.configs.BitsAndBytes import BitsAndBytesConfig


class Llama2DataSource(ChatBotDataSource):

    def __init__(self,
                 bnb_config: BitsAndBytesConfig == None
                 ) -> None:
        l2hf = Llama2Hugginface()
        bnb_config = BitsAndBytesConfig

        if bnb_config != None:
            llm_model = l2hf.model_quantize(bnb_config)
        else:
            raise Exception("full model needs to be implemented")
        
        self._hf_pipeline = l2hf.pipeline_from_pretrained_model(llm_model)
        self._2hf = l2hf
        self._llm_model = llm_model
        return None
    
    def generate_base_answer(self,
                        question: str) -> ChatBotReadModel:
        prompt = self._2hf.langchain_prompt()
        question_formatted = prompt.format(user_message=question)
        answer = self._hf_pipeline(question_formatted)
        return answer

