import torch
from typing import List
from transformers import LlamaForCausalLM, PreTrainedTokenizer, AutoTokenizer
from glob import glob
from tqdm import tqdm
import re
from fire import Fire
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def aggregate_initial_attention_layers(attn, n_layers=1):
    '''Extract average attention vector'''
    avged = []
    for layer in attn[:n_layers]:
        layer_attns = layer.cpu().squeeze(0)
        attns_per_head = layer_attns.mean(dim=0)

        # in case the generation is produced by beam search, also average over beams
        # todo: need to use attention indices to filter instead of just averaging
        if len(attns_per_head.shape) == 3:
            attns_per_head = attns_per_head.mean(dim=0)

        vec = torch.concat((
            # We zero the first entry because it's what's called
            # null attention (https://aclanthology.org/W19-4808.pdf)
            torch.tensor([0.]),
            # usually there's only one item in attns_per_head but
            # on the first generation, there's a row for each token
            # in the prompt as well, so take [-1]
            attns_per_head[-1][1:],
            # add zero for the final generated token, which never
            # gets any attention
            torch.tensor([0.]),
        ))
        avged.append(vec / vec.sum())
    return torch.stack(avged).mean(dim=0)



def default_aggregate_attention(attn):
    '''Extract average attention vector'''
    avged = []
    for layer in attn:
        layer_attns = layer.cpu().squeeze(0)
        attns_per_head = layer_attns.mean(dim=0)

        # in case the generation is produced by beam search, also average over beams
        # todo: need to use attention indices to filter instead of just averaging
        if len(attns_per_head.shape) == 3:
            attns_per_head = attns_per_head.mean(dim=0)

        vec = torch.concat((
            # We zero the first entry because it's what's called
            # null attention (https://aclanthology.org/W19-4808.pdf)
            torch.tensor([0.]),
            # usually there's only one item in attns_per_head but
            # on the first generation, there's a row for each token
            # in the prompt as well, so take [-1]
            attns_per_head[-1][1:],
            # add zero for the final generated token, which never
            # gets any attention
            torch.tensor([0.]),
        ))
        avged.append(vec / vec.sum())
    return torch.stack(avged).mean(dim=0)


def find_sublist_start(larger: List[int], smaller: List[int]):
    if len(smaller) > len(larger):
        raise ValueError("sublist cannot be longer than reference list")

    if len(smaller) == 0 or len(larger) == 0:
        raise ValueError("cannot find sublist in empty list")

    for i in range(len(larger) - len(smaller) + 1):
        if larger[i:i+len(smaller)] == smaller:
            return i

    return -1


def longest_common_subarray_in_first(arr1, arr2):
    n, m = len(arr1), len(arr2)
    # Create a 2D table to store lengths of longest common suffixes of substrings.
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # Length of the longest common subarray
    longest_length = 0

    # To store the ending index of the longest common subarray in arr1
    end_index_arr1 = -1

    # Building the DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if arr1[i - 1] == arr2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > longest_length:
                    longest_length = dp[i][j]
                    end_index_arr1 = i - 1

    # If no common subarray found, return None
    if longest_length == 0:
        return None

    # Calculate the start index based on the length of the longest common subarray
    start_index_arr1 = end_index_arr1 - longest_length + 1

    return (start_index_arr1, end_index_arr1)


def heterogenous_stack(vecs):
    '''Pad vectors with zeros then stack'''
    max_length = max(v.shape[0] for v in vecs)
    return torch.stack([
        torch.concat((v, torch.zeros(max_length - v.shape[0])))
        for v in vecs
    ])


def get_average_attention_over_sequence(aggregated_attention, token_ids: torch.Tensor, sequence: str,
                                        tokenizer: PreTrainedTokenizer):
    # check if sequence is available in token_ids
    seq_token_ids = tokenizer(sequence, add_special_tokens=False, return_tensors='pt')['input_ids'][0].tolist()

    # to avoid initial token mismatch due to space or newline
    # seq_token_ids = seq_token_ids[2:]

    # prompt_token_ids = token_ids[:len(token_ids)-len(aggregated_attention)]

    # attn_m = heterogenous_stack([
    #     torch.tensor([
    #         1 if i == j else 0
    #         for j, token in enumerate(prompt_token_ids)
    #     ])
    #     for i, token in enumerate(prompt_token_ids)
    # ] + aggregated_attention)
    attn_m = heterogenous_stack(aggregated_attention)

    # beg_idx = find_sublist_start(token_ids.tolist(), seq_token_ids)
    #
    beg_idx, end_idx = longest_common_subarray_in_first(token_ids.tolist(), seq_token_ids)

    # print("beg_idx: ", beg_idx)
    # print("sequence: ", sequence)
    # print("seq_token_ids: ", seq_token_ids)
    # print("token_ids: ", token_ids)

    if beg_idx == -1:
        raise ValueError(f"sequence {seq_token_ids} not found in reference token_ids: {token_ids}")

    # avg_attn_score = attn_m[-len(aggregated_attention):, beg_idx:beg_idx+len(seq_token_ids)].mean().item()
    avg_attn_score = attn_m[:, beg_idx:end_idx].mean().item()
    return avg_attn_score


def get_attention_per_strategy(files, tokenizer):
    data = []
    for file in files:
        with open(file, 'rb') as f:
            data.append(pickle.load(f))

    res = {}
    for example in tqdm(data):
        if example is None:
            continue
        for strategy, (tokens, attentions) in example.items():
            strategy_str = f'"{strategy}"'
            attn = get_average_attention_over_sequence(attentions, tokens, sequence=strategy_str, tokenizer=tokenizer)
            if strategy in res:
                res[strategy].append(attn)
            else:
                res[strategy] = [attn]
    return res


def get_attention_per_strategy_and_description(files, tokenizer):
    data = []
    for file in files:
        with open(file, 'rb') as f:
            data.append(pickle.load(f))

    res = {}
    for example in tqdm(data):
        if example is None:
            continue
        for strategy, (tokens, attentions) in example.items():
            decoded_prompt = tokenizer.decode(tokens, skip_special_tokens=False)
            # print(decoded_prompt)
            match = re.findall(f'You are a helpful and caring friend.+{strategy} strategy\.', decoded_prompt)

            if len(match) == 0 or len(match) > 1:
                print("WARNING: no match or multiple matches found for prompt: ", decoded_prompt)
                continue

            attn = get_average_attention_over_sequence(attentions, tokens, sequence=match[0], tokenizer=tokenizer)
            if strategy in res:
                res[strategy].append(attn)
            else:
                res[strategy] = [attn]
    return res


def get_comparison_barchart(attn1, attn2, name1, name2, save_dir):
    attn1_avg = {k: np.mean(v) for k, v in attn1.items()}
    attn2_avg = {k: np.mean(v) for k, v in attn2.items()}

    xlabels = list(attn1_avg.keys()) + list(attn2_avg.keys())
    yvals = list(attn1_avg.values()) + list(attn2_avg.values())
    legends = [name1] * len(attn1_avg) + [name2] * len(attn2_avg)

    df = pd.DataFrame({'strategy': xlabels, 'attention': yvals, 'prompt_type': legends})
    sns.barplot(df, x='strategy', y='attention', hue='prompt_type')
    plt.xticks(rotation=90)
    plt.savefig(f'{save_dir}.png')


def compare_prompting_methods(output_dir1, output_dir2, model_name_or_path, name1='exp1', name2='exp2',
                              save_dir='outputs/comparison', cache_dir=None):

    files1 = glob(f'{output_dir1}/*.pkl')
    files2 = glob(f'{output_dir2}/*.pkl')

    print(f"Found {len(files1)} files in {output_dir1}")
    print(f"Found {len(files2)} files in {output_dir2}")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    attn1 = get_attention_per_strategy(files1, tokenizer)
    attn2 = get_attention_per_strategy(files2, tokenizer)

    get_comparison_barchart(attn1, attn2, name1, name2, save_dir)

    attn1 = get_attention_per_strategy_and_description(files1, tokenizer)
    attn2 = get_attention_per_strategy_and_description(files2, tokenizer)

    get_comparison_barchart(attn1, attn2, name1, name2, save_dir + '_with_description')


if __name__ == '__main__':
    Fire(compare_prompting_methods)

