
from core.llm.models.llama2.Llama2Huggingface import Llama2Hugginface, Llama2Prompt
from features.chatbot.data.datasource.api.ChatBotDataSource import ChatBotDataSource
from features.chatbot.data.datasource.api.VectorDbSource import VectorDbSource
from features.chatbot.data.models.ChatBotModel import ChatBotReadModel
from core.llm.models.configs.BitsAndBytes import BitsAndBytesConfig
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import ConversationChain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate

class Llama2DataSource(ChatBotDataSource):

    def __init__(self,
                 vector_db: VectorDbSource,
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
        self._langchain_hf_pipeline = HuggingFacePipeline(pipeline=self._hf_pipeline)
        self._basic_prompt = PromptTemplate.from_template(Llama2Prompt.prompt_template)
        self._chatchain_prompt = PromptTemplate.from_template(Llama2Prompt.chatchain_prompt_template)

        #todo, bound memory to with user
        #for chat
        self._chat_chain_memory = ConversationBufferMemory(ai_prefix="AI Agent:")
        self._chat_chain = ConversationChain(llm=self._langchain_hf_pipeline,
                               prompt=self._chatchain_prompt,
                               verbose=False,
                               memory=self._chat_chain_memory
                              )

        #for rag
        self._chat_rag_history = []
        self._vector_db = vector_db
        self._conversational_rag_chain: ConversationalRetrievalChain = ConversationalRetrievalChain.from_llm(
              self._langchain_hf_pipeline,
              vector_db.retriever(),
              return_source_documents=False
        )

        return None
    
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

    def chat_rag(self,
                question: str,
                get_history:bool = False) -> ChatBotReadModel:
        
        question_formatted = self._basic_prompt.format(user_message=question)
        retrieval_qa_format = {"question": question_formatted,
                       "chat_history": self._chat_rag_history}
        answer = self._conversational_rag_chain(retrieval_qa_format)

        self._chat_rag_history.append((question,answer["answer"]))

        response_history = self._chat_rag_history if get_history else []

        cbrm = ChatBotReadModel(question=question,
                                model_use=self._l2hf.model_id,
                                answer=answer["answer"],
                                chat_history=response_history)
        return cbrm
    
    def chat(self, question: str) -> ChatBotReadModel:
        """interactive chat using langchain ConverstationChain class.
        You will be able to pose question to the model initialized by
        this class

        Args:
            question (str): question to ask

        Returns:
            ChatBotReadModel: response chatbot model
        """
        answer = self._chat_chain.predict(input=question)
        cbrm =  ChatBotReadModel(question=question,
                                model_use=self._l2hf.model_id,
                                answer=answer,
                                )
        return cbrm