import json

from torch.cuda import device
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Dict, Tuple
import fire
import torch
import os
from tqdm import tqdm
from accelerate import Accelerator
import random
from glob import glob


random.seed(11335577)



def convert_to_llama2_chat_format(sys_msg: str, conversations: List[str], tokenizer, **kwargs) -> str:
    messages = [{'role': 'system', 'content': sys_msg}]
    for i in range(0, len(conversations) - 1, 2):
        messages.append({'role': 'user', 'content': conversations[i]})
        messages.append({'role': 'assistant', 'content': conversations[i + 1]})
    messages.append({'role': 'user', 'content': conversations[-1]})

    formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    return formatted_prompt


def get_model_and_tokenizer(model_name, cache_dir, load_in_4bit=True):
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=False, load_in_4bit=True
        )
        # Copy the model to each device
        device_map = (
            {"": Accelerator().local_process_index}
        )
        torch_dtype = torch.bfloat16
    else:
        device_map = None
        quantization_config = None
        torch_dtype = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
        #todo: make sure flash attention is installed
        # attn_implementation="flash_attention_2"
    )
    if not load_in_4bit:
        model = model.to(device)

    model.eval()

    return model, tokenizer

def run(data_dir='../prompting/outputs/', output_path='./outputs/out.json', model_name_or_path='', load_in_4bit=True,
        max_new_tokens=1024, cache_dir=None):
    all_files = glob(os.path.join(data_dir, '*.json'))
    model, tokenizer = get_model_and_tokenizer(model_name_or_path, cache_dir, load_in_4bit)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for fpath in tqdm(all_files):
        with open(fpath, 'r') as f:
            data = json.load(f)

        input_ids = convert_to_llama2_chat_format(
            "You are a helpful, precise and accurate emotional support expert",
            data['dialog'],
            tokenizer
        )
        input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(device)
        outputs = model.generate(input_ids, do_sample=False, max_new_tokens=max_new_tokens, temperature=0.7)
        response = outputs[0][len(input_ids[0]):]
        output_txt = tokenizer.decode(response, skip_special_tokens=True).strip()
        data['responses']['unconditional'] = output_txt

        results.append(data)
        with open(output_path, 'w') as f:
            for res in results:
                f.write(json.dumps(res) + '\n')

if __name__ == '__main__':
    fire.Fire(run)

