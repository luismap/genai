import transformers
from torch import bfloat16

class BitsAndBytesConfig:
    def get_configs():
        #to do, make get it as kwargs
        bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16)

        return bnb_config