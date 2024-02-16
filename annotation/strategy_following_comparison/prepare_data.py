from glob import glob
import fire
import json
from typing import List, Tuple, Dict, Union
from uuid import uuid4
from annotation.strategy_following_comparison.metadata import strategy_metadata
import numpy as np
import pickle
from transformers import AutoTokenizer
from prompting.analyze_attention_weights import get_average_attention_over_sequence

#
# This annotation task is for comparing two model's responses to the same
# prompt given different strategies.

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-70b-chat-hf')


def load_models_responses(model1_file: str, model2_file: str) -> Dict:
    """Assumes the files are for the same prompt"""

    with open(model1_file, 'r') as f:
        model1_data = json.load(f)
    with open(model2_file, 'r') as f:
        model2_data = json.load(f)

    conv_history = model1_data['dialog']
    assert conv_history == model2_data['dialog'], "Different conversation histories found!"

    speakers = model1_data['speakers']
    assert speakers == model2_data['speakers'], "Different speakers found!"

    model1_strategy_responses: Dict = model1_data['responses']
    model2_strategy_responses: Dict = model2_data['responses']

    assert model1_strategy_responses.keys() == model2_strategy_responses.keys(), "Different strategies found!"

    return {
        'history': conv_history,
        'speakers': speakers,
        'model1_responses': model1_strategy_responses,
        'model2_responses': model2_strategy_responses
    }


def calculate_example_SRAs(attention_file_path: str) -> Union[Dict[str, float], None]:
    with open(attention_file_path, 'rb') as f:
        attention_data = pickle.load(f)

    # in case the file is empty. will be filtered when preparing the annotation data
    if attention_data is None:
        return

    sra_dict = {}
    for strategy, (tokens, attentions) in attention_data.items():
        strategy_str = f'"{strategy}"'
        sra_dict[strategy] = get_average_attention_over_sequence(attentions, tokens, sequence=strategy_str,
                                                                 tokenizer=tokenizer)

    return sra_dict




def prepare_comparison_data(model1_output_dir: str, model2_output_dir: str, output_file: str):
    model1_files = glob(f"{model1_output_dir}/*.json")
    model2_files = glob(f"{model2_output_dir}/*.json")

    model_1_file_names = [file.split('/')[-1] for file in model1_files]
    model_2_file_names = [file.split('/')[-1] for file in model2_files]

    shared_sample_file_names = set(model_1_file_names).intersection(set(model_2_file_names))

    model1_files = [f"{model1_output_dir}/{file}" for file in shared_sample_file_names]
    model2_files = [f"{model2_output_dir}/{file}" for file in shared_sample_file_names]

    all_data = []
    for model1_file, model2_file in zip(model1_files, model2_files):
        model1_attn_file = model1_file.replace('.json', '_attentions.pkl')
        model2_attn_file = model2_file.replace('.json', '_attentions.pkl')

        model1_sra = calculate_example_SRAs(model1_attn_file)
        model2_sra = calculate_example_SRAs(model2_attn_file)

        if model1_sra is None or model2_sra is None:
            continue

        sra_comp = {}
        for strategy in model1_sra.keys():
            sra_comp[strategy] = [model1_sra[strategy], model2_sra[strategy]]

        d = load_models_responses(model1_file, model2_file)
        d['file_names'] = [model1_file, model2_file]
        d['conv_id'] = str(uuid4())
        d['sra_comparison'] = sra_comp

        all_data.append(d)

    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=4)


def postprocess_response(response: str, max_len=512) -> str:
    seq_txt = response.lower().strip()

    # Remove the assistant prefix
    if 'assistant:' in seq_txt:
        seq_txt = seq_txt.split('assistant:')[-1]

    if len(seq_txt) > max_len:
        return seq_txt.strip()[:max_len]+"..."
    return seq_txt.strip()


def postprocess_instance_for_label_studio(instance: Dict) -> Dict:
    history = instance['history']
    speakers = instance['speakers']

    model1_response = instance['model1_response']
    model2_response = instance['model2_response']

    history_txt = ""
    for speaker, utt in zip(speakers, history):
        history_txt += f"<br><b>{speaker}</b>: {utt.strip()}"

    instance['history'] = history_txt
    instance['model1_response'] = postprocess_response(model1_response)
    instance['model2_response'] = postprocess_response(model2_response)
    instance['metadata'] = f"Strategy: {instance['strategy']}<br><br>{instance['metadata']}"

    return instance

def prepare_annotation_data(comparison_data_file: str, output_file: str, max_samples=100):
    """
    Prepare the data for annotation. This will include the conversation history, the responses from the two models,
    and the unique id for each conversation and the corresponding strategy.
    this function makes sure we sample strategies uniformly across the dataset and have correct sample distribution
    between 3 annotators. We also add additional metadata about the strategy that the models need to follow.
    """
    with open(comparison_data_file, 'r') as f:
        comparison_data = json.load(f)

    strategies = list(strategy_metadata.keys())

    # loop over conversations and pick strategy with the least number of picks
    strategy_occurrences = {strategy: 0 for strategy in strategies}

    samples = []

    for conv_data in comparison_data:
        cur_strategies = list(conv_data['model1_responses'].keys())
        if len(cur_strategies) == 0:
            continue

        cur_strategy_cnts = [strategy_occurrences[strategy] for strategy in cur_strategies]
        min_strategy = cur_strategies[cur_strategy_cnts.index(min(cur_strategy_cnts))]
        strategy_occurrences[min_strategy] += 1

        responses = [conv_data['model1_responses'][min_strategy], conv_data['model2_responses'][min_strategy]]
        file_names = conv_data['file_names']
        sras = conv_data['sra_comparison'][min_strategy]

        # randomly select first and second response
        first_resp_id = np.random.randint(0, 2)

        first_resp = responses[first_resp_id]
        second_resp = responses[1-first_resp_id]

        samples.append({
            'metadata': strategy_metadata[min_strategy],
            'conv_id': conv_data['conv_id']+f"_{min_strategy}",
            'file_names': [file_names[first_resp_id], file_names[1-first_resp_id]],
            'history': conv_data['history'],
            'model1_response': first_resp,
            'model2_response': second_resp,
            'model1_sra': sras[first_resp_id],
            'model2_sra': sras[1-first_resp_id],
            'strategy': min_strategy,
            'speakers': conv_data['speakers']
        })

    print(f"Total samples: {len(samples)}")
    print(f"Strategy occurrences: {strategy_occurrences}")

    # todo: configure it based on platform
    samples = [postprocess_instance_for_label_studio(instance) for instance in samples]
    np.random.shuffle(samples)
    samples = samples[:max_samples]

    # save the data
    with open(output_file, 'w') as f:
        json.dump(samples, f, indent=4)


def run(mode: str, model1_output_dir: str = '.', model2_output_dir: str = '.', output_file: str = '.',
        comparison_data_file: str = '.', max_samples: int = 100):
    if mode == 'prepare_comparison_data':
        prepare_comparison_data(model1_output_dir, model2_output_dir, output_file)
    elif mode == 'prepare_annotation_data':
        prepare_annotation_data(comparison_data_file, output_file, max_samples)
    else:
        raise ValueError("Invalid mode!")


if __name__ == '__main__':
    fire.Fire(run)
