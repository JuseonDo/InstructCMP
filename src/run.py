import fire
import torch
from tqdm import  tqdm
from typing import List,Dict,Tuple
from detokenize.detokenizer import detokenize
from transformers import BitsAndBytesConfig
from transformers import(
    AutoTokenizer,
    LlamaForCausalLM,
)

from src.inference_utils import inference
from src.utils import get_template, apply_template

def main(
        model_size:str = "13",
        batch_size:int = 10,
):  
    model_name = f"meta-llama/Llama-2-{model_size}b-chat-hf"

    print("Loading Model...")
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = LlamaForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=nf4_config,
        )
    model.config.pad_token_id = model.config.eos_token_id
    model.eval()

    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              use_fast=True,
                                              padding_side="left",
                                              )
    tokenizer.pad_token = tokenizer.eos_token

    template = get_template()
    instances = [
        {
            "src": "sentence",
            "del_len": 10
        }
    ]
    prompts = apply_template(instances, template)

    generated_text = inference(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        batch_size=batch_size,
        max_new_tokens=200,
        do_sample=False,
    )

if __name__ == '__main__':
    with torch.no_grad():
        fire.Fire(main)