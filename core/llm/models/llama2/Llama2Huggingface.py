from transformers.models.llama.configuration_llama import LlamaConfig
from transformers import LlamaTokenizerFast, AutoModelForCausalLM, PreTrainedModel
from transformers import pipeline, AutoConfig, AutoTokenizer
from core.llm.models.configs.BitsAndBytes import BitsAndBytesConfig
from core.utils.Configs import Settings
from huggingface_hub import login
from torch import float16
from langchain import PromptTemplate
from langchain_community.llms import VLLM


class Llama2Prompt:
    prompt_template = """<s>[INST] <<SYS>>
your are a good and helpful assistant. Help me with my questions. If you do not know the answer, please do not make up the answers.
<</SYS>>

{user_message} [/INST]
"""

    chatchain_prompt_template = """<s>[INST] <<SYS>>
your are a good and helpful assistant. Help me with my questions. If you do not know the answer, please do not make up the answers.
<</SYS>>

# Current conversation:
# {history}
# Human: {input} [/INST]
"""


class Llama2Hugginface:
    def __init__(
        self, prompt: str = "", model_id: str = "meta-llama/Llama-2-7b-chat-hf"
    ) -> None:
        """class to abstract interaction with llama models
        Args:
            model_id (str, optional): Defaults to "meta-llama/Llama-2-7b-chat-hf".
        """
        self.model_id = model_id
        self.settings = Settings()
        if prompt == "":
            self._prompt_template = Llama2Prompt.prompt_template
        else:
            self._prompt_template = prompt
        login(self.settings.hf_token)

        return None

    @property
    def prompt(self):
        return self._prompt_template

    @prompt.setter
    def prompt(self, value: str):
        self._prompt_template = value

    def langchain_prompt(self) -> PromptTemplate:
        """returns a prompt template that can be use
        with langchain

        Returns:
            PromptTemplate: the prompt template
        """
        return PromptTemplate.from_template(self.prompt)

    def model_config(self) -> LlamaConfig:
        """returns configuration for the llama model

        Returns:
            LlamaConfig: _description_
        """
        config = AutoConfig.from_pretrained(
            self.model_id, auth_token=self.settings.hf_token
        )

        return config

    def tokenizer(self) -> LlamaTokenizerFast:
        """tokenizer used by this llama model

        Returns:
            LlamaTokenizerFast: _description_
        """
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, use_auth_token=self.settings.hf_token
        )
        # tokenizer.pad_token = "[PAD]"
        # tokenizer.padding_side = "right"
        return tokenizer

    def model_quantize(self, bitandsbitesconfig: BitsAndBytesConfig):
        """return a quantize llama2 model base on the
        passed bitsandbites configurations

        Args:
            bitandsbitesconfig (BitsAndBytesConfig): _description_

        Returns:
            _type_: _description_
        """
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            config=self.model_config(),
            quantization_config=bitandsbitesconfig.get_configs(),
            device_map="auto",  # TODO check for custom device map
            use_auth_token=self.settings.hf_token,
        )
        return model

    def model(self):
        """returns llama2 model, with the current model_id of this class.
        Returns:
            _type_: _description_
        """
        # TODO test it, we need to have at least the same amount of gpu vram as what the
        # current model asks for
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            config=self.model_config(),
            device_map="auto",  # TODO check for custom device map
            use_auth_token=self.settings.hf_token,
        )
        return model

    def pipeline_from_pretrained_model(
        self,
        model: PreTrainedModel,
        task: str = "text-generation",
        temperature: float = 0.1,
        max_new_tokens: int = 512,
        repetition_penalty: float = 1.1,
        full_text: bool = True,
        batch_size: int = 1,
        device: str = "auto",
    ):
        """given a custom pretrained model, create a huggingface
        pipeline.

        Args:
            model (PreTrainedModel): the model to be added to the pipeline
            task (str, optional): task. Defaults to "text-generation".
            temperature (float, optional): temperature. Defaults to 0.1.
            max_new_tokens (int, optional): max tokens. Defaults to 512.
            repetition_penalty (float, optional): repetition penalty. Defaults to 1.1.

        Returns:
            pipeline: huggingface pipeline
        """
        tokenizer = self.tokenizer()
        if tokenizer.pad_token is None:
            print("updating tokenizer because of None")
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))
            model.config.pad_token = tokenizer.pad_token

        pline = pipeline(
            model=model,
            tokenizer=tokenizer,
            return_full_text=full_text,  # langchain expects the full text
            task=task,
            # batch_size=batch_size,
            # we pass model parameters here too
            temperature=temperature,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
            max_new_tokens=max_new_tokens,  # max number of tokens to generate in the output
            repetition_penalty=repetition_penalty,  # without this output begins repeating
            device_map=device,
            eos_token_id=tokenizer.eos_token_id,
        )
        return pline

    def pipeline(self, task: str = "text-generation", device: str = "auto"):
        """return a base huggingface pipeline
        for the current model.
        Will get the model that is mapped as default by huggingface pipeline
        for the current model_id
        Args:
            task (str, optional): Defaults to "text-generation".

        Returns:
            _type_: _description_
        """
        huggingface_pipeline = pipeline(
            task, model=self.model_id, torch_dtype=float16, device_map=device
        )  # TODO check for custom device map
        return huggingface_pipeline

    def langchain_vllm_model(
        self,
        max_new_tokens: int = 512,
        top_k: int = 10,
        top_p: float = 0.95,
        temperature: float = 0.2,
        tensor_parallel_size: int = 1,
        dtype: str = "half",
    ):  # find the return type and add it
        """
        Given the model id, return a vllm version of it.
        """
        model = VLLM(
            model=self.model_id,
            trust_remote_code=True,  # mandatory for hf models
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
        )

        return model
