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


def generate_resp(model, tokenizer, examples, device, temperature=0.1, top_p=0.9, max_new_tokens=1024, do_sample=True):

    prompts = [[
        {"role": "system", "content": "todo system message"},
        {"role": "user", "content": ex['query']},
    ] for ex in examples]

    input_chat_prompts = [tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        tokenize=False,
    ) for prompt in prompts]

    inputs = tokenizer(input_chat_prompts, return_tensors='pt', padding=True, truncation=False)

    terminators = [
        tokenizer.eos_token_id,
        #todo: add whatever token is necessary
        # tokenizer.convert_tokens_to_ids("<|eot_id|>")
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


def generate_responses(test_path: str = 'data/test.json', model_path: str = 'outputs/', save_batch_size: int = 10,
                       save_path: str = "data/results.json", model_response_col: str = "model_response", qlora=False):

    test_data = []
    with open(test_path, 'r') as f:
        for line in f:
            ex = json.loads(line)
            if 'query' in ex:
                test_data.append({'query': ex['query'], 'response': ex['response'], 'id': ex['id']})
            elif 'messages' in ex:
                test_data.append(ex)
            else:
                raise ValueError(f"Invalid example: {ex}")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("total validation data:", len(test_data))

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
        )
    else:
        #todo: set correct data type for the model you're testing
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        model = model.to(device)

    lora_tokenizer = AutoTokenizer.from_pretrained(model_path)
    lora_tokenizer.padding_side = 'left'
    lora_tokenizer.pad_token = lora_tokenizer.eos_token
    lora_tokenizer.pad_token_id = lora_tokenizer.convert_tokens_to_ids(lora_tokenizer.eos_token_id)

    model.eval()

    result = []
    i = 0
    def checkpoint(data, save_path):
        with open(save_path, 'w') as f:
            for d in data:
                f.write(json.dumps(d) + '\n')

    pbar = tqdm(total=len(test_data))
    while i < len(test_data):
        examples = test_data[i:i+save_batch_size]
        resps = generate_resp(model, lora_tokenizer, examples, device)

        for ex, resp in zip(examples, resps):
            res_json = ex.copy()
            res_json[model_response_col] = resp
            result.append(res_json)

        if len(result) % save_batch_size == 0:
            checkpoint(result, save_path)

        i += save_batch_size
        pbar.update(save_batch_size)

    checkpoint(result, save_path)


if __name__ == '__main__':
    Fire(generate_responses)