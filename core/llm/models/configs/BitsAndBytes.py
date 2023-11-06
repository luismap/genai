import transformers
from torch import bfloat16

class BitsAndBytesConfig:
    """wrapper class to interact with bitsandbytes configs
    for quantization
    """
    def __init__(self,
                 configs: transformers.BitsAndBytesConfig = None) -> None:
        self._configs = configs
        return None

    def get_configs(self):
        """get current bitsandbytes configs.
        if class was initialized with configs, then we get those
        otherwise, we get the default configs.

        Returns:
            BitsAndBytesConfig: the configs
        """
        #to do, make get it as kwargs
        if self._configs != None:
            return self._configs

        bnb_4bit_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16)

        return bnb_4bit_config