from transformers.models.llama.configuration_llama import LlamaConfig
from transformers import LlamaTokenizerFast, AutoModelForCausalLM, PreTrainedModel
from transformers import pipeline, AutoConfig, AutoTokenizer
from core.llm.models.configs.BitsAndBytes import BitsAndBytesConfig
from core.utils.Configs import Settings
from huggingface_hub import login
from torch import float16
from langchain import PromptTemplate

class Llama2Prompt:
    prompt_template = """<s>[INST] <<SYS>>
your are a good and helpful assistant. Help me with my questions. If you do not know the answer, please do not make up the answers.
<</SYS>>

{user_message} [/INST]
"""

class Llama2Hugginface:
    def __init__(self,
                 prompt: str = "",
                 model_id: str ="meta-llama/Llama-2-7b-chat-hf"
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
            self.model_id,
            auth_token=self.settings.hf_token)
        
        return config
    
    def tokenizer(self) -> LlamaTokenizerFast:
        """tokenizer used by this llama model

        Returns:
            LlamaTokenizerFast: _description_
        """
        tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, 
                use_auth_token=self.settings.hf_token)
        return tokenizer
    
    def model_quantize(self,
                       bitandsbitesconfig: BitsAndBytesConfig):
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
        device_map='cuda:0', #TODO check for custom device map
        use_auth_token=self.settings.hf_token)
        return model

    def pipeline_from_pretrained_model(self,
                            model: PreTrainedModel,
                            task: str = "text-generation",
                            temperature: float = 0.1,
                            max_new_tokens: int = 512,
                            repetition_penalty: float = 1.1
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
        pline =  pipeline(model=model, 
                            tokenizer=self.tokenizer(),
                            return_full_text=True,  # langchain expects the full text
                            task=task,
                            # we pass model parameters here too
                            temperature=temperature,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
                            max_new_tokens=max_new_tokens,  # max number of tokens to generate in the output
                            repetition_penalty=repetition_penalty  # without this output begins repeating
        )
        return pline
    
    def pipeline(self,
                 task: str = "text-generation"):
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
        task,
        model=self.model_id,
        torch_dtype=float16,
        device_map='cuda:0') #TODO check for custom device map
        return huggingface_pipeline