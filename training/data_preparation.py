from typing import List, Dict
from fire import Fire
import json
from copy import copy
from tqdm import tqdm
from prompting.llama_prompt import modified_extes_support_strategies


def build_sys_msg_from_strategy(strategy: str, situation: str) -> str:
    description = modified_extes_support_strategies[strategy]

    return """You are a helpful and caring AI which is an expert in emotional support.\
 A user has come to you with the following situation: "{situation}".
 Use "{cur_strategy}" strategy ({strategy_description}) for answering the user.\
 make your response short and to the point.""".format(cur_strategy=strategy, strategy_description=description,
                                                      situation=situation)


def convert_conversations_to_chat_format(data: List[Dict]) -> List[Dict]:
    processed_convs = []

    for ex in tqdm(data):
        messages = []
        speakers = ['user' if s == 'seeker' else 'assistant' for s in ex['speakers']]
        turns = [t.strip() for t in ex['dialog']]

        for s, t in zip(speakers, turns):
            messages.append({'role': s, 'content': t})

        for label, resp in ex['responses'].items():
            cur_msgs = copy(messages)
            cur_msgs.append({'role': 'assistant', 'content': resp})

            sys_msg = build_sys_msg_from_strategy(label)
            cur_msgs.insert(0, {'role': 'system', 'content': sys_msg})
            processed_convs.append({'messages': cur_msgs})

    return processed_convs

def run(extended_ds_path: str, output_path: str) -> None:

    with open(extended_ds_path, 'r') as f:
        data = json.load(f)

    print("total number of conversations: ", len(data))
    #todo: add id and split data across conversations

    processed_convs = convert_conversations_to_chat_format(data)

    print("total number of samples: ", len(processed_convs))

    #todo: add id and splits
    with open(output_path, 'w') as f:
        for ex in processed_convs:
            f.write(json.dumps(ex) + '\n')


if __name__ == '__main__':
    Fire(run)