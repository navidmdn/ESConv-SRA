
from fire import Fire
import json
from tqdm import tqdm
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSequenceClassification,
    PreTrainedModel, PreTrainedTokenizer)
import torch
from typing import List, Dict, Tuple
import numpy as np
from prompting.analyze_attention_weights import (default_aggregate_attention,
                                                 get_average_attention_over_sequence)


np.random.seed(43)

def predict_strategy_classes(sequences: List[str], classifier: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> List[str]:
    all_preds = []
    inputs = tokenizer(sequences, padding=True, truncation=True, max_length=512, return_tensors="pt")
    inputs = {k: v.to(classifier.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = classifier(**inputs)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
    id2label = {v: k for k, v in classifier.config.label2id.items()}
    all_preds.extend([id2label[p.item()] for p in preds])
    return all_preds


def convert_to_llama_chat_partial_conv_format(messages: List[Dict], tokenizer,
                                               n_turns_as_conv=3, history_first=True, **kwargs) -> str:
    assert messages[0]['role'] == 'system', 'First message should be system message!'

    if messages[-1]['role'] == 'user':
        prompt = messages
    else:
        prompt = messages[:-1]

    if n_turns_as_conv % 2 != 1:
        raise ValueError("n_turns_as_conv should be odd number")

    # not considering system message
    history_len = len(prompt) - 1
    system_msg = prompt[0]['content']

    if history_len > n_turns_as_conv:
        turns_in_history = [p['content'] for p in prompt[1:-n_turns_as_conv]]

        conv_history_str = "conversation history:\n\n"

        for i in range(0, len(turns_in_history), 2):
            conv_history_str += "User: " + turns_in_history[i].strip() + "\n"
            conv_history_str += "Assistant: " + turns_in_history[i + 1].strip() + "\n"

        if history_first:
            updated_sys_msg = f"{conv_history_str}\n\n{system_msg.strip()}"
        else:
            updated_sys_msg = f"{system_msg.strip()}\n\n{conv_history_str}"

        prompt[0] = {'role': 'system', 'content': updated_sys_msg}
        prompt = [prompt[0]] + prompt[-n_turns_as_conv:]

    formatted_prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    return formatted_prompt


def convert_to_llama_chat_format(messages: List[Dict], tokenizer, **kwargs) -> str:
    assert messages[0]['role'] == 'system', 'First message should be system message!'

    if messages[-1]['role'] == 'user':
        prompt = messages
    else:
        prompt = messages[:-1]

    formatted_prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    return formatted_prompt


def generate_resp(model, tokenizer, examples, device, base_model_type, baseline_name,
    temperature=0.1, top_p=0.9, max_new_tokens=1024, do_sample=True, n_turns_as_conv=3, history_first=True,
    get_sra=False):

    if base_model_type in ['llama2', 'llama3']:
        if baseline_name in ['c1_hf', 'c3_hf', 'c5_hf', 'c1_hl', 'c3_hl', 'c5_hl']:

            if 'c1' in baseline_name:
                n_turns_as_conv = 1
            elif 'c3' in baseline_name:
                n_turns_as_conv = 3
            elif 'c5' in baseline_name:
                n_turns_as_conv = 5
            else:
                raise ValueError(f"Unsupported baseline name: {baseline_name}")

            if 'hf' in baseline_name:
                history_first = True
            elif 'hl' in baseline_name:
                history_first = False
            else:
                raise ValueError(f"Unsupported baseline name: {baseline_name}")

            prompt_constructor = convert_to_llama_chat_partial_conv_format
        elif baseline_name == 'standard':
            prompt_constructor = convert_to_llama_chat_format
        else:
            raise ValueError(f"Unsupported baseline name: {baseline_name}")
    else:
        raise ValueError(f"Unsupported base model type: {base_model_type}")


    prompts = [prompt_constructor(ex['messages'], tokenizer, n_turns_as_conv=n_turns_as_conv,
                                  history_first=history_first) for ex in examples]

    inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=False)


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
        return_dict_in_generate=True,
        output_attentions=get_sra
    )

    sra_vals = None
    if get_sra:
        # attention_spans = [ex['messages'][0]['content'] for ex in examples]
        attention_spans = [f'\"{ex["strategy"]}\" strategy' for ex in examples]
        agg_attentions = [[default_aggregate_attention(t).cpu() for t in attentions] for attentions in output.attentions]
        sra_vals = [get_average_attention_over_sequence(agg_attention, token_ids=sequence, sequence=attention_span, tokenizer=tokenizer) for
                    sequence, agg_attention, attention_span in zip(output.sequences, agg_attentions, attention_spans)]

    resps = tokenizer.batch_decode(output.sequences[:, input_ids.shape[1]:].cpu().numpy(), skip_special_tokens=True)
    return resps, sra_vals


def generate_responses(test_path: str = 'data/test_processed.json', model_path: str = 'meta-llama/Llama-2-7b-chat-hf',
                       save_batch_size: int = 10, infere_strategy=False, strategy_classifer_path=None,
                       save_path: str = "data/results.json", model_response_col: str = "model_response", qlora=False,
                       baseline_name='c3_hl', base_model_type='llama2', cache_dir=None,
                       max_test_samples=None, get_sra=False):

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

    np.random.shuffle(test_data)
    test_data = test_data[:max_test_samples] if max_test_samples is not None else test_data

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
            cache_dir=cache_dir
        )
    else:

        #todo: set correct data type for the model you're testing
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, cache_dir=cache_dir)
        model = model.to(device)

    lora_tokenizer = AutoTokenizer.from_pretrained(model_path)
    lora_tokenizer.padding_side = 'left'
    lora_tokenizer.pad_token = lora_tokenizer.eos_token
    lora_tokenizer.pad_token_id = lora_tokenizer.convert_tokens_to_ids(lora_tokenizer.eos_token)

    model.eval()

    result = []
    i = 0
    def checkpoint(data, save_path):
        with open(save_path, 'w') as f:
            for d in data:
                f.write(json.dumps(d) + '\n')

    if infere_strategy:
        classifer_model = AutoModelForSequenceClassification.from_pretrained(strategy_classifer_path, cache_dir=cache_dir)
        classifer_tokenizer = AutoTokenizer.from_pretrained(strategy_classifer_path, cache_dir=cache_dir)
        classifer_model = classifer_model.to(device)
        classifer_model.eval()

    pbar = tqdm(total=len(test_data))
    while i < len(test_data):
        examples = test_data[i:i+save_batch_size]
        resps, sras = generate_resp(model, lora_tokenizer, examples, device,
                              base_model_type=base_model_type, baseline_name=baseline_name,
                              get_sra=get_sra)
        if infere_strategy:
            inferred_classes = predict_strategy_classes(resps, classifer_model, classifer_tokenizer)

        for k, (ex, resp) in enumerate(zip(examples, resps)):
            res_json = ex.copy()
            res_json[model_response_col] = resp
            result.append(res_json)

            if infere_strategy:
                res_json['inferred_strategy'] = inferred_classes[k]

            if get_sra:
                res_json['sra'] = sras[k]

        if len(result) % save_batch_size == 0:
            checkpoint(result, save_path)

        i += save_batch_size
        pbar.update(save_batch_size)

    checkpoint(result, save_path)


if __name__ == '__main__':
    Fire(generate_responses)