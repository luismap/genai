from features.chatbot.data.datasource.ChatBotLlama2DataSource import Llama2DataSource
from core.llm.models.configs.BitsAndBytes import BitsAndBytesConfig
import pytest
import os


@pytest.fixture(scope="module")
def use_vllm_env_var_false() -> pytest.fixture:
    # using fixtures now, if we want to use many env vars
    # we can use monkeypatch
    os.environ["use_vllm"] = "false"
    return os.environ["use_vllm"]


@pytest.fixture
def llama2_datasource_hugginface(use_vllm_env_var_false):
    bnb_config = BitsAndBytesConfig()
    l2hf = Llama2DataSource(bnb_config=bnb_config)
    yield l2hf
    # teardown phase
    del l2hf


@pytest.fixture(scope="module")
def use_vllm_env_var_true() -> pytest.fixture:
    os.environ["use_vllm"] = "true"
    return os.environ["use_vllm"]


@pytest.fixture
def llama2_datasource_vllm(use_vllm_env_var_true):
    l2ds = Llama2DataSource()
    yield l2ds
    del l2ds


class TestChatbotLlama2DatasourceHuggingfaceIntegration:
    def test_can_use_hugging_face(self, llama2_datasource_hugginface):
        l2hf = llama2_datasource_hugginface
        question = "hi!"
        assert len(l2hf.generate_base_answer(question).answer) >= 2


class TestChatbotLlama2DatasourceVllmIntegration:
    def test_can_use_vllm(self, llama2_datasource_vllm):
        l2ds = llama2_datasource_vllm
        question = "hi!"
        assert len(l2ds.generate_base_answer(question).answer) >= 2
