
from typing import List, Tuple
from core.llm.models.llama2.Llama2Huggingface import Llama2Hugginface, Llama2Prompt
from features.chatbot.data.datasource.api.ChatBotDataSource import ChatBotDataSource
from features.chatbot.data.models.ChatBotModel import ChatBotReadModel, ChatBotPayloadModel
from core.llm.models.configs.BitsAndBytes import BitsAndBytesConfig
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate

class MyLlama2DsPrompt:
    memory_prompt_template = """<s>[INST] <<SYS>> your are a good and helpful assistant. Help me with my questions. If you do not know the answer, please do not make up the answers.
<</SYS>>
{history}
 {input} [/INST]
"""

class Llama2DataSource(ChatBotDataSource):
    """
    wrapper class for interacting with `Llama2` models from `huggingface`.

    Args:
        ChatBotDataSource : abstract class which decides main interactions
    """

    def __init__(self
                 ,bnb_config: BitsAndBytesConfig = None
                 ,device: str = "auto"
                 ) -> None:
        l2hf = Llama2Hugginface()
        
        if bnb_config != None:
            llm_model = l2hf.model_quantize(bnb_config)
        else:
            raise Exception("full model needs to be implemented")

        self._hf_pipeline = l2hf.pipeline_from_pretrained_model(llm_model
                                                                ,full_text=False
                                                                ,device=device
                                                                )
        self._l2hf = l2hf
        self._llm_model = llm_model
        #self._langchain_hf_pipeline = HuggingFacePipeline(pipeline=self._hf_pipeline, batch_size=12)
        self._basic_prompt = PromptTemplate.from_template(Llama2Prompt.prompt_template)
        self._memory_prompt = PromptTemplate.from_template(MyLlama2DsPrompt.memory_prompt_template)
        #self._chatchain_prompt = PromptTemplate.from_template(Llama2Prompt.chatchain_prompt_template)
        #todo, bound memory to with user
        #for chat
        chat_memory = []
        self._user_info = {"default":
                                {"get_history":"false",
                                 "history": chat_memory}
                                }

        self._users = {"default"}

    def _add_user(self, user_name: str):
        if user_name not in self._users:
            chat_memory = []
            self._user_info[user_name] = {"get_history":"false",
                                "history":chat_memory}
                                
            self._users.add(user_name)
            return True
        else:
            return False
        

    def is_available(self) -> bool:
        #need to be tweaked for fallbacks
        return True

    def generate_base_answer(self,
                        question: str) -> ChatBotReadModel:
        prompt = self._l2hf.langchain_prompt()
        question_formatted = prompt.format(user_message=question)
        answer = self._hf_pipeline(question_formatted)
        answer_top_1 = answer[0]["generated_text"] #can be tweaked for more answers

        cbrm = ChatBotReadModel(question=question,
                                model_use=self._l2hf.model_id,
                                answer=answer_top_1)
        return cbrm

    def _get_history(self) -> List[Tuple[str,str]]:
        messages = self._chat_chain_memory.chat_memory.messages
        messages_parsed = [(messages[i].content, messages[i+1].content)for i in [i for i in range(0,len(messages),2)]]
        return messages_parsed
    
    def clean_user_history(self, user_id: str) -> bool:
        if user_id in self._users:
            self._user_info[user_id]["history"] = []
            return True
        else: #TODO add exception 
            return False
        
    
    def _generate_history(self,history: List[Tuple[str, str]]) -> str:
        """
        given a list of tuples with values question and answer
        generate a str for use as history for llama model

        Args:
            history (List[Tuple[str, str]]): history tuple

        Returns:
            str: history formatted
        """
        combined = [ q + " [/INST] " + a + "</s><s>[INST]" for q,a in history]
        return "\n".join(combined)
    
    def chat(self, cbpms = List[ChatBotPayloadModel]) -> List[ChatBotReadModel]:
        data = []
        model_response = []

        for cb_payload in cbpms:
            user_id = cb_payload.user_id
            if user_id not in self._users:
                self._add_user(user_id)

            user_hist = self._user_info[user_id]["history"]
            history_g = self._generate_history(user_hist)
            question_formatted = self._memory_prompt.format(
                history=history_g,
                input = cb_payload.question)
            data.append(question_formatted)

        responses = zip(self._hf_pipeline(data),cbpms)

        for res,payload in responses:
            response_history = self._generate_history(self._user_info[payload.user_id]["history"]) if payload.history else ""
            answer = res[0]["generated_text"]
            cbrm =  ChatBotReadModel(user_id=payload.user_id,
                                question=payload.question,
                                model_use=self._l2hf.model_id,
                                answer=answer,
                                batch_history= response_history
                                )
            model_response.append(cbrm)
            self._user_info[payload.user_id]["history"].append(
                (payload.question,answer)
            )

        return model_response