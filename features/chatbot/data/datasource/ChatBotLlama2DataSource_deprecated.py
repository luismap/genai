
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
     memory_prompt_template = """ <<SYS>> your are a good and helpful assistant. Help me with my questions. If you do not know the answer, please do not make up the answers.
<</SYS>>
{history}
[INST] {input} [/INST]
"""

class Llama2DataSource(ChatBotDataSource):
    """
    wrapper class for interacting with `Llama2` models from `huggingface`.

    Args:
        ChatBotDataSource : abstract class which decides main interactions
    """

    def __init__(self,
                 bnb_config: BitsAndBytesConfig = None
                 ) -> None:
        l2hf = Llama2Hugginface()
        
        if bnb_config != None:
            llm_model = l2hf.model_quantize(bnb_config)
        else:
            raise Exception("full model needs to be implemented")

        self._hf_pipeline = l2hf.pipeline_from_pretrained_model(llm_model)
        self._l2hf = l2hf
        self._llm_model = llm_model
        #self._langchain_hf_pipeline = HuggingFacePipeline(pipeline=self._hf_pipeline, batch_size=12)
        self._basic_prompt = PromptTemplate.from_template(Llama2Prompt.prompt_template)
        #self._chatchain_prompt = PromptTemplate.from_template(Llama2Prompt.chatchain_prompt_template)
        #todo, bound memory to with user
        #for chat
        chat_chain_memory = ConversationBufferMemory(ai_prefix="AI Agent:")
        chat_chain = ConversationChain(llm=self._langchain_hf_pipeline,
                               prompt=self._chatchain_prompt,
                               verbose=False,
                               memory=chat_chain_memory
                              )
        self._user_to_model = {"default":
                                {"history":"false",
                                "chat_model":chat_chain}
                                }

        self._users = {"default"}

    def _add_user(self, user_name: str):
        if user_name not in self._users:
            chat_chain_memory = ConversationBufferMemory(ai_prefix="AI Agent:")
            chat_chain = ConversationChain(llm=self._langchain_hf_pipeline,
                                prompt=self._chatchain_prompt,
                                verbose=False,
                                memory=chat_chain_memory
                                )
            self._user_dict = {user_name:
                               {"history":"false",
                                "chat_model":chat_chain}
                                }
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
    
    def clean_memory(self) -> bool:
        self._chat_chain_memory.clear()
        return True
    
    def chat(self, cbpms = List[ChatBotPayloadModel]) -> List[ChatBotReadModel]:
        """
        interactive chat using langchain `ConversationChain` class.
        You will be able to pose question to the model initialized by
        this class

        Args:
            question (str): question to ask

        Returns:
            ChatBotReadModel: response chatbot model
        """
        data = []
        model_response = []

        for cb_payload in cbpms:
            data.append(cb_payload.question)

        #currently will just use the default user
        responses = zip(self._user_to_model["default"]["chat_model"].batch(data),cbpms)

        for res,payload in responses:
            response_history = res["history"] if payload.history == "true" else []
            cbrm =  ChatBotReadModel(question=res["input"],
                                model_use=self._l2hf.model_id,
                                answer=res["response"],
                                batch_history= response_history
                                )
            model_response.append(cbrm)

        return model_response