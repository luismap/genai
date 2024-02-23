from typing import List
from core.llm.models.llama2.Llama2Huggingface import Llama2Hugginface
from core.utils.Configs import Settings
from features.chatbot.data.datasource.api.RagChatBotDataSource import (
    RagChatBotDataSource,
)
from features.chatbot.data.datasource.api.VectorDbSource import VectorDbSource
from core.llm.models.configs.BitsAndBytes import BitsAndBytesConfig
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain import PromptTemplate
from features.chatbot.data.models.ChatRagModel import (
    ChatRagPayloadModel,
    SourceDocument,
)
from features.chatbot.data.models.ChatRagModel import ChatRagReadModel
import yaml
import logging
import logging.config
from core.utils.MyUtils import MyUtils
from core.utils.text.TextLlmUtils import TextLlmUtils


class RagLlama2DataSource(RagChatBotDataSource):
    """
    wrapper class for interacting with `Llama2` models from `huggingface`.

    Args:
        RagChatBotDataSource : abstract class which decides main interactions
    """

    def __init__(
        self,
        vector_db: VectorDbSource,
        bnb_config: BitsAndBytesConfig = None,
        device: str = "auto",
        vllm_configs: dict = {"tensor_parallel_size": 2},
    ) -> None:
        l2hf = Llama2Hugginface()

        with open("logging.yaml", "rt") as f:
            config = yaml.safe_load(f.read())
            logging.config.dictConfig(config)

        appProps = MyUtils.load_properties("general")["app"]
        self._logger = logging.getLogger(appProps["logger"])
        self._logger.info("initializing llama2 llm for rag")
        l2hf = Llama2Hugginface()
        settings = Settings()

        self._use_vllm = settings.use_vllm

        if self._use_vllm:
            self._vllm_model = l2hf.langchain_vllm_model(**vllm_configs)
            self._logger.info("using vllm on llama2 llm")

        else:
            if bnb_config is not None:
                llm_model = l2hf.model_quantize(bnb_config)
                self._logger.info("using hugging face transformers")
            else:
                raise Exception("full model needs to be implemented")

            self._llm_model = llm_model
            self._hf_pipeline = l2hf.pipeline_from_pretrained_model(
                llm_model, device=device, full_text=False
            )

            self._langchain_hf_pipeline = HuggingFacePipeline(
                pipeline=self._hf_pipeline
            )

        self._l2hf = l2hf
        chatchain_prompt_template = """
<s>[INST]<<SYS>>You are a helpful agent.You will be given a context with information 
about different topics and a history of our current conversation. Please answer the questions 
using the information provided in the context and history. If you do not know the answer, do 
not make up the answers. Please be short and concise with your answer.<</SYS>>

{context}

# Current conversation:
# {history}
# question: {question} [/INST]
"""
        prompt_template = """<s>[INST]<<SYS>>
You are a helpful agent.You will be given a context with information about different topics.
Please answer the questions using that context. If you do not know the answer, do not make up the answers.
<</SYS>>
{user_message} [/INST]
"""
        self._basic_prompt = PromptTemplate.from_template(prompt_template)
        self._chatchain_prompt = PromptTemplate.from_template(chatchain_prompt_template)

        # for rag

        self._vector_db = vector_db

        if self._use_vllm:
            llm = self._vllm_model
        else:
            llm = self._langchain_hf_pipeline

        self._llm = llm

        """
        self._conversational_rag_chain: ConversationalRetrievalChain = (
            ConversationalRetrievalChain.from_llm(
                llm,
                vector_db.retriever(),
                return_source_documents=True,
                rephrase_question=False,
            )
        )
        """
        chat_rag_history = []
        self._user_info = {
            "default": {
                "get_history": "false",
                "history": chat_rag_history,
                "vector_db": vector_db,
            }
        }

        self._users = {"default"}

    def _add_user(self, user_name: str):
        if user_name not in self._users:
            chat_rag_history = []
            self._user_info[user_name] = {
                "get_history": "false",
                "history": chat_rag_history,
                "vector_db": self._vector_db.create_new_instance(),
            }

            self._users.add(user_name)
            return True
        else:
            return False

    def clean_user_history(self, user_id: str) -> bool:
        if user_id not in self._users:
            return False  # add exception
        else:
            self._user_info[user_id]["history"] = []
            self._user_info[user_id][
                "vector_db"
            ] = self._vector_db.create_new_instance()
            return True

    def is_available(self) -> bool:
        # need to be tweaked for fallbacks
        return True

    def generate_base_answer(self, question: str) -> ChatRagReadModel:
        prompt = self._l2hf.langchain_prompt()
        question_formatted = prompt.format(user_message=question)

        if self._use_vllm:
            answer = self._vllm_model.invoke(question_formatted)
        else:
            answer = self._hf_pipeline(question_formatted)
            answer = answer[0]["generated_text"]  # can be tweaked for more answers
        source_documents = []

        cbrm = ChatRagReadModel(
            user_id="default",
            question=question,
            model_use=self._l2hf.model_id,
            answer=answer,
            source_doc=source_documents,
        )
        return cbrm

    def chat_rag(
        self, chatrag_models=List[ChatRagPayloadModel]
    ) -> List[ChatRagReadModel]:
        users = []
        questions = []
        docs = []
        for crm in chatrag_models:
            if crm.user_id not in self._users:
                self._add_user(user_name=crm.user_id)

            retrieve_docs = self._user_info[crm.user_id]["vector_db"].similarity_search(
                crm.question
            )
            context = TextLlmUtils.format_docs(retrieve_docs)
            chat_history = self._user_info[crm.user_id]["history"]
            question_formatted = self._chatchain_prompt.format(
                context=context, history=chat_history, question=crm.question
            )
            users.append(crm)
            questions.append(question_formatted)
            docs.append(retrieve_docs)

        if self._use_vllm:
            answers = self._vllm_model.batch(questions)
        else:
            answers = self._langchain_hf_pipeline.batch(questions)

        cbrms = []

        for user, ans, docs in zip(users, answers, docs):
            response_history = (
                self._user_info[user.user_id]["history"] if user.history else []
            )
            self._user_info[user.user_id]["history"].append((user.question, ans))

            source_docs = set()
            for doc in docs:
                try:
                    title = doc.metadata["title"]
                except KeyError:
                    title = "llm brain"

                try:
                    source = doc.metadata["source"]
                except KeyError:
                    source = self._l2hf.model_id

                source_docs.add((title, source))

            source_docs = [
                SourceDocument(title=title, source=source)
                for title, source in source_docs
            ]

            cbrm = ChatRagReadModel(
                user_id=user.user_id,
                question=user.question,
                model_use=self._l2hf.model_id,
                answer=ans,
                chat_history=response_history,
                source_doc=source_docs,
            )
            cbrms.append(cbrm)

        return cbrms
