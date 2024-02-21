from typing import List
from core.llm.models.llama2.Llama2Huggingface import Llama2Hugginface, Llama2Prompt
from features.chatbot.data.datasource.api.RagChatBotDataSource import (
    RagChatBotDataSource,
)
from features.chatbot.data.datasource.api.VectorDbSource import VectorDbSource
from core.llm.models.configs.BitsAndBytes import BitsAndBytesConfig
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain import PromptTemplate
from features.chatbot.data.models.ChatRagModel import (
    ChatRagPayloadModel,
    SourceDocument,
)

from features.chatbot.data.models.ChatRagModel import ChatRagReadModel


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
    ) -> None:
        l2hf = Llama2Hugginface()

        if bnb_config != None:
            llm_model = l2hf.model_quantize(bnb_config)
        else:
            raise Exception("full model needs to be implemented")

        self._hf_pipeline = l2hf.pipeline_from_pretrained_model(
            llm_model, device=device
        )
        self._l2hf = l2hf
        self._llm_model = llm_model
        self._langchain_hf_pipeline = HuggingFacePipeline(pipeline=self._hf_pipeline)
        chatchain_prompt_template = """<s>[INST]<<SYS>>
You are a helpful agent.You will be given a context with information about different topics.
Please answer the questions using that context. If you do not know the answer, do not make up the answers.
<</SYS>>

# Current conversation:
# {history}
# Human: {input} [/INST]
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
        self._conversational_rag_chain: ConversationalRetrievalChain = (
            ConversationalRetrievalChain.from_llm(
                self._langchain_hf_pipeline,
                vector_db.retriever(),
                return_source_documents=True,
                rephrase_question=False,
            )
        )
        chat_rag_history = []
        self._user_info = {
            "default": {"get_history": "false", "history": chat_rag_history}
        }

        self._users = {"default"}

    def _add_user(self, user_name: str):
        if user_name not in self._users:
            chat_rag_history = []
            self._user_info[user_name] = {
                "get_history": "false",
                "history": chat_rag_history,
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
            return True

    def is_available(self) -> bool:
        # need to be tweaked for fallbacks
        return True

    def generate_base_answer(self, question: str) -> ChatRagReadModel:
        prompt = self._l2hf.langchain_prompt()
        question_formatted = prompt.format(user_message=question)
        answer = self._hf_pipeline(question_formatted)
        answer_top_1 = answer[0]["generated_text"]  # can be tweaked for more answers

        cbrm = ChatRagReadModel(
            question=question, model_use=self._l2hf.model_id, answer=answer_top_1
        )
        return cbrm

    def chat_rag(
        self, chatrag_models=List[ChatRagPayloadModel]
    ) -> List[ChatRagReadModel]:
        users = []
        questions = []
        for crm in chatrag_models:
            if crm.user_id not in self._users:
                self._add_user(user_name=crm.user_id)

            question_formatted = self._basic_prompt.format(user_message=crm.question)

            chat_history = self._user_info[crm.user_id]["history"]
            retrieval_qa_format = {
                "question": question_formatted,
                "chat_history": chat_history,
            }
            users.append(crm)
            questions.append(retrieval_qa_format)

        answers = self._conversational_rag_chain.batch(questions)

        cbrms = []

        for user, ans in zip(users, answers):
            response_history = (
                self._user_info[user.user_id]["history"] if user.history else []
            )
            self._user_info[user.user_id]["history"].append(
                (user.question, ans["answer"])
            )

            source_docs = set()
            for doc in ans["source_documents"]:
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
                answer=ans["answer"],
                chat_history=response_history,
                source_doc=source_docs,
            )
            cbrms.append(cbrm)

        return cbrms
