"""
This script is used to generate responses from a model using the Hugging Face library. It is used to generate responses
from either a model on HF or a fine-tuned model. The script takes in a test file, model path, and other parameters to
generate responses from the model. The responses are then saved to a file.

Note that you need to set model specific configs like the tokenizer, model, and other parameters in the script.
"""

from fire import Fire
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from prompting.llama_prompt import modified_extes_support_strategies


def turn_print(role, msg, strategy=None):
    strategy_string = strategy if strategy is not None else ""
    print(f"=================== {role}-{strategy_string} ===================")
    print(msg)
    print(f"=================================================================")

def get_sys_msg_with_strategy(strategy):
    if strategy == None:
        return """You are a helpful, precise and accurate emotional support expert"""

    description = modified_extes_support_strategies[strategy]

    return """You are a helpful and caring AI which is an expert in emotional support.\
 A user has come to you with emotional challenges, distress or anxiety.\
 Use "{cur_strategy}" strategy ({strategy_description}) for answering the user.\
 make your response short and to the point.""".format(cur_strategy=strategy, strategy_description=description)


def generate_resp(model, tokenizer, sys_msg, examples, device, temperature=0.8, top_p=0.9,
                  max_new_tokens=1024, do_sample=True):

    prompts = [[
        {"role": "system", "content": sys_msg},
        *ex['messages'],
    ] for ex in examples]

    print(prompts[0])
    input_chat_prompts = [tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        tokenize=False,
    ) for prompt in prompts]

    inputs = tokenizer(input_chat_prompts, return_tensors='pt', padding=True, truncation=False)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    input_ids = inputs['input_ids'].to(device)
    attention_masks = inputs['attention_mask'].to(device)

    output = model.generate(
        input_ids,
        attention_mask=attention_masks,
        max_new_tokens=max_new_tokens,
        eos_token_id=terminators,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
    )

    resps = tokenizer.batch_decode(output[:, input_ids.shape[1]:].cpu().numpy(), skip_special_tokens=True)
    return resps

def get_strategy():
    strategies = list(modified_extes_support_strategies.keys())
    for i, strategy in enumerate(strategies):
        print(f"{i + 1}: {strategy}")
    print(f"{len(strategies) + 1}: No strategy")

    strategy_id = input("Choose a strategy: ")

    if strategy_id == str(len(strategies) + 1):
        strategy = None
    else:
        strategy = strategies[int(strategy_id) - 1]

    return strategy


def generate_responses(model_path: str = 'outputs/', qlora=False, cache_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if qlora:
        torch_dtype = torch.bfloat16
        quant_storage_dtype = torch.bfloat16
        bnb_4bit_use_double_quant = True
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_storage=quant_storage_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            torch_dtype=quant_storage_dtype,
            cache_dir=cache_dir
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, cache_dir=cache_dir)
        model = model.to(device)

    lora_tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    lora_tokenizer.padding_side = 'left'
    lora_tokenizer.pad_token = lora_tokenizer.eos_token
    # lora_tokenizer.pad_token_id = lora_tokenizer.eos_token_id

    model.eval()

    user_msg = "Hello!"
    strategy = None
    sys_msg = get_sys_msg_with_strategy(strategy)
    messages = [{"role": "user", "content": user_msg}]

    while True:
        if user_msg == "reset":
            print("restarting chat...")
            user_msg = input("User: ")
            strategy = get_strategy()
            sys_msg = get_sys_msg_with_strategy(strategy)
            messages = [{"role": "user", "content": user_msg}]

        turn_print("user", user_msg)

        assistant_msg = generate_resp(model, lora_tokenizer, sys_msg,
                                      [{"messages": messages}], device)[0]

        turn_print("assistant", assistant_msg, strategy)
        messages.append({"role": "assistant", "content": assistant_msg})

        strategy = get_strategy()

        sys_msg = get_sys_msg_with_strategy(strategy)

        user_msg = input("User: ")
        messages.append({"role": "user", "content": user_msg})






if __name__ == '__main__':
    Fire(generate_responses)