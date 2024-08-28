import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Dict, Tuple
import fire
from prompting.llama_prompt import modified_extes_support_strategies
from prompting.llama_prompt import B_SYS, B_INST, E_INST, E_SYS
import torch
import os
from tqdm import tqdm
from accelerate import Accelerator
import random
import pickle
import time
from prompting.analyze_attention_weights import default_aggregate_attention


random.seed(11335577)


template1 = """You are a helpful, precise and accurate emotional support expert.\
 The user has come to you with the following situation: "{situation}". continue the\
 conversation for one turn using "{cur_strategy}" strategy. {strategy_description} make your response short and to the point.\
 Do not provide additional info. only respond in one paragraph that satisfies {cur_strategy} strategy."""

template2 = """You are a helpful and caring friend.\
 Your best friend has come to you with the following situation: "{situation}". continue the\
 conversation for one turn using "{cur_strategy}" strategy. {strategy_description} make your response short and natural.\
 Do not provide additional information. only respond in one paragraph that satisfies {cur_strategy} strategy."""

template3 = """You are a helpful and caring friend.\
 Your best friend has come to you with the following situation: "{situation}". continue the\
 conversation for one turn using "{cur_strategy}" strategy. {strategy_description} make your response short and to the point.\
 Do not provide additional info. only respond in one or two short sentences that satisfies {cur_strategy} strategy."""

template4 = """You are a helpful and caring friend.\
 Your best friend has come to you with the following situation: "{situation}". continue the\
 conversation for one turn using "{cur_strategy}" strategy ({strategy_description}) make your response short and to the point.\
 Do not provide additional info. only respond in one paragraph that satisfies {cur_strategy} strategy.\
 answer in this format: assistant: <response>"""

template5 = """You are a helpful and caring friend.\
 Your best friend has come to you with the following situation: "{situation}". continue the\
 conversation for one turn using "{cur_strategy}" strategy ({strategy_description}) make your response short and to the point.\
 Do not provide additional info. only respond in ONE SENTENCE in this format: assistant: <one sentence response>"""


def load_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]


class ESPromptOutput:
    def __init__(self, dialog: List[str], situation: str, speakers: List[str], responses: Dict[str, str],
                 prompts: Dict[str, str], attentions: Dict[str, Tuple] = None):
        self.dialog = dialog
        self.situation = situation
        self.speakers = speakers
        self.responses = responses
        self.prompts = prompts
        self.attentions = attentions

    def to_dict(self):
        return {
            "dialog": self.dialog,
            "situation": self.situation,
            "speakers": self.speakers,
            "responses": self.responses,
            "prompts": self.prompts,
        }

    def __repr__(self):
        respones_str = ""
        for strategy, resp in self.responses.items():
            respones_str += f"{'*' * 300}\n\n strategy: {strategy}\n\n response: {resp}\n\n{'*' * 300}\n\n"
        return "_______________________".join([
            f"Situation: {self.situation}",
            f"Dialog: {self.dialog}",
            f"responses: {respones_str}"
        ])


def convert_to_llama2_chat_format_manually(sys_msg: str, conversations: List[str]) -> str:
    """
        <s>[INST] <<SYS>>
    {{ system_prompt }}
    <</SYS>>

    {{ user_msg_1 }} [/INST] {{ model_answer_1 }} </s><s>[INST] {{ user_msg_2 }} [/INST]
    """
    conv_0 = conversations[0]
    conversations = conversations[1:]

    result = f"<s>{B_INST} {B_SYS}{sys_msg}{E_SYS} {conv_0} {E_INST} "
    i = 0
    while i < len(conversations):
        ai_msg = conversations[i]
        human_msg = conversations[i + 1]
        result += f"{ai_msg} </s><s>{B_INST} {human_msg} {E_INST} "
        i += 2

    return result


def convert_to_llama2_chat_partial_conv_format(sys_msg: str, conversations: List[str], tokenizer,
                                               n_turns_as_conv=3, history_first=True, **kwargs) -> str:
    if n_turns_as_conv % 2 != 1:
        raise ValueError("n_turns_as_conv should be odd number")

    conv_messages = []
    for i in range(max(len(conversations) - n_turns_as_conv, 0), len(conversations) - 1, 2):
        conv_messages.append({'role': 'user', 'content': conversations[i].strip()})
        conv_messages.append({'role': 'assistant', 'content': conversations[i + 1].strip()})
    conv_messages.append({'role': 'user', 'content': conversations[-1].strip()})

    if len(conversations) > n_turns_as_conv:
        conversations = conversations[:-n_turns_as_conv]

    conv_history_str = "conversation history:\n"
    for i in range(0, len(conversations) - 1, 2):
        conv_history_str += "user: " + conversations[i].strip() + "\n"
        conv_history_str += "assistant: " + conversations[i + 1].strip() + "\n"

    if history_first:
        sys_msg = f"{conv_history_str}\n{sys_msg.strip()}"
    else:
        sys_msg = f"{sys_msg.strip()}\n{conv_history_str}"

    messages = [{'role': 'system', 'content': sys_msg}]
    messages.extend(conv_messages)

    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    return formatted_prompt

def convert_to_mistral_chat_partial_conv_format(sys_msg: str, conversations: List[str], tokenizer,
                                               n_turns_as_conv=3, history_first=True, **kwargs) -> str:
    if n_turns_as_conv % 2 != 1:
        raise ValueError("n_turns_as_conv should be odd number")

    conv_messages = []
    for i in range(max(len(conversations) - n_turns_as_conv, 0), len(conversations) - 1, 2):
        conv_messages.append({'role': 'user', 'content': conversations[i].strip()})
        conv_messages.append({'role': 'assistant', 'content': conversations[i + 1].strip()})
    conv_messages.append({'role': 'user', 'content': conversations[-1].strip()})

    if len(conversations) > n_turns_as_conv:
        conversations = conversations[:-n_turns_as_conv]

    conv_history_str = "conversation history:\n"
    for i in range(0, len(conversations) - 1, 2):
        conv_history_str += "user: " + conversations[i].strip() + "\n"
        conv_history_str += "assistant: " + conversations[i + 1].strip() + "\n"

    if history_first:
        sys_msg = f"{conv_history_str}\n{sys_msg.strip()}"
    else:
        sys_msg = f"{sys_msg.strip()}\n{conv_history_str}"

    assert conv_messages[0]['role'] == 'user'
    conv_messages[0]['content'] = f"{sys_msg}\n\n{conv_messages[0]['content']}"

    formatted_prompt = tokenizer.apply_chat_template(conv_messages, tokenize=False)
    return formatted_prompt


def convert_to_llama2_chat_format(sys_msg: str, conversations: List[str], tokenizer, **kwargs) -> str:
    messages = [{'role': 'system', 'content': sys_msg}]
    for i in range(0, len(conversations) - 1, 2):
        messages.append({'role': 'user', 'content': conversations[i]})
        messages.append({'role': 'assistant', 'content': conversations[i + 1]})
    messages.append({'role': 'user', 'content': conversations[-1]})

    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    return formatted_prompt


def convert_to_mistral_chat_format(sys_msg: str, conversations: List[str], tokenizer, **kwargs) -> str:
    messages = []
    for i in range(0, len(conversations) - 1, 2):
        messages.append({'role': 'user', 'content': f"{sys_msg}\n\n{conversations[i]}"})
        messages.append({'role': 'assistant', 'content': conversations[i + 1]})
    messages.append({'role': 'user', 'content': conversations[-1]})

    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    return formatted_prompt


def convert_to_llama2_llm_format(sys_msg: str, conversations: List[str], tokenizer) -> str:
    formatted_prompt = f"{tokenizer.bos_token}{sys_msg}\n\n"

    for i in range(0, len(conversations) - 1, 2):
        formatted_prompt += f"seeker: {conversations[i].strip()}\n"
        formatted_prompt += f"supporter: {conversations[i + 1].strip()}\n"
    formatted_prompt += f"seeker: {conversations[-1].strip()}\n"
    formatted_prompt += f"supporter: "
    return formatted_prompt


def get_model_and_tokenizer(model_name, cache_dir, load_in_4bit=True):
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

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

    return model, tokenizer


def get_continuation_prompt(conversation, model, tokenizer, model_type='llama', get_attentions=False,
                            max_new_tokens=512, prompt_constructor=convert_to_llama2_chat_format, sample_prob=0.3,
                            history_first=True, n_turns_as_conv=3):
    dialog = conversation['dialog_history']
    speakers = conversation['prev_speakers']
    situation = conversation['situation']

    if speakers[0] == 'supporter':
        speakers = speakers[1:]
        dialog = dialog[1:]

    assert speakers[0] == 'seeker'
    assert speakers[-1] == 'seeker'
    responses = {}
    prompts = {}
    attentions = {}

    for strategy, desc in tqdm(modified_extes_support_strategies.items()):

        if random.random() > sample_prob:
            continue
        sys_msg = template4.format(situation=situation, cur_strategy=strategy, strategy_description=desc)

        if model_type == 'llama' or model_type == 'mistral':
            prompt = prompt_constructor(sys_msg, dialog, tokenizer, n_turns_as_conv=n_turns_as_conv, history_first=history_first)
            input_ids = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)['input_ids'].to(model.device)
            # print("prompt: ", prompt)
            if len(input_ids[0]) > 1400:
                print(f"PROMPT LENGTH ({len(input_ids[0])}) exceeds the memory limit. skipping this instance")
                continue

        elif model_type == 'mistral':
            pass
        else:
            raise ValueError(f"model_type should be one of ['llama', 'mistral'], but got {model_type}")

        #todo: currently aggregation only supports greedy decoding
        outputs = model.generate(input_ids, do_sample=False,
                                 output_attentions=get_attentions, max_new_tokens=max_new_tokens,
                                 return_dict_in_generate=True)
        prompts[strategy] = prompt

        print(outputs.attentions, type(outputs.attentions))
        if get_attentions:
            agg_attentions = [default_aggregate_attention(t).cpu() for t in outputs.attentions]
            # last tokens attentions should be the size of full sequence
            assert agg_attentions[-1].shape[0] == len(
                outputs.sequences[0]), f"{agg_attentions[-1].shape[0]} != {len(outputs.sequences[0])}"
            attentions[strategy] = (outputs.sequences[0].cpu(), agg_attentions)

        response = outputs.sequences[0][len(input_ids[0]):]
        output_txt = tokenizer.decode(response, skip_special_tokens=True).strip()
        responses[strategy] = output_txt
        print("\nprompt: ", prompt)
        print(f"\n\nstrategy:\n{strategy}\n\nresponse:\n{output_txt}")

    if attentions:
        return ESPromptOutput(dialog=dialog, situation=situation, speakers=speakers, responses=responses,
                              prompts=prompts, attentions=attentions)
    res = ESPromptOutput(dialog=dialog, situation=situation, speakers=speakers, responses=responses, prompts=prompts)
    return res


def run(data_path='../esconv/conversations.json', min_turn=3, max_turn=10, model_path='nickypro/tinyllama-15M',
        cache_dir=None, output_path='./outputs', load_in_4bit=True, get_attentions=False, max_new_tokens=512,
        n_iters=-1, prompt_constructor='partial', n_turns_as_conv=None, history_first=None, sample_prob=0.3):
    data = load_jsonl(data_path)
    data = [d for d in data if min_turn <= d['turn'] <= max_turn]

    model, tokenizer = get_model_and_tokenizer(model_path, cache_dir, load_in_4bit)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token

    if 'llama' in model_path:
        if prompt_constructor == 'partial':
            assert n_turns_as_conv is not None
            assert history_first is not None
            prompt_constructor_func = convert_to_llama2_chat_partial_conv_format
        elif prompt_constructor == 'full':
            prompt_constructor_func = convert_to_llama2_chat_format
        else:
            raise ValueError(f"prompt_constructor should be one of ['partial', 'full'], but got {prompt_constructor}")

        print(f"using prompt constructor: {prompt_constructor}")
    elif 'mistral' in model_path:
        if prompt_constructor == 'partial':
            assert n_turns_as_conv is not None
            assert history_first is not None
            prompt_constructor_func = convert_to_mistral_chat_partial_conv_format
        elif prompt_constructor == 'full':
            prompt_constructor_func = convert_to_mistral_chat_format
        else:
            raise ValueError(f"prompt_constructor should be one of ['partial', 'full'], but got {prompt_constructor}")

        print(f"using prompt constructor: {prompt_constructor}")

    os.makedirs(output_path, exist_ok=True)

    i = 0
    if n_iters == -1:
        n_iters = len(data)

    while i < len(data):
        if i >= n_iters:
            break

        rand_id = random.randint(0, len(data)-1)
        if os.path.exists(os.path.join(output_path, f'{rand_id}.json')):
            i += 1
            continue

        conversation = data[rand_id]
        generated_conts = get_continuation_prompt(
            conversation, model, tokenizer,
            get_attentions=get_attentions,
            max_new_tokens=max_new_tokens,
            prompt_constructor=prompt_constructor_func,
            history_first=history_first,
            n_turns_as_conv=n_turns_as_conv,
            sample_prob=sample_prob
        )

        with open(os.path.join(output_path, f'{rand_id}.json'), 'w') as f:
            json.dump(generated_conts.to_dict(), f)

        if get_attentions:
            with open(os.path.join(output_path, f'{rand_id}_attentions.pkl'), 'wb') as f:
                pickle.dump(generated_conts.attentions, f)

        i += 1


if __name__ == '__main__':
    fire.Fire(run)
