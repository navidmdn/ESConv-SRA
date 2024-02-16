import pickle
import fire
from typing import List, Tuple
import json
from sentence_transformers import SentenceTransformer
import os
import glob
from tqdm import tqdm
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Encoding on device:", DEVICE)


def encode_responses(model, responses: List[str]):
    return model.encode(responses, convert_to_tensor=False, batch_size=32, show_progress_bar=True, device=DEVICE)


def postprocess_response(response: str, strategy: str) -> str:
    seq_txt = response.lower().strip()

    # Remove the assistant prefix
    if 'assistant:' in seq_txt:
        seq_txt = seq_txt.split('assistant:')[-1]
    # Remove exact strategy mentions
    seq_txt = seq_txt.replace(strategy.lower(), '')

    return seq_txt.strip()


def load_strategy_and_response(response_file: str) -> Tuple[List[str], List[str]]:
    with open(response_file, 'r') as f:
        data = json.load(f)

    responses = data['responses']
    return list(responses.keys()), list(responses.values())


def run(sbert_model_name='all-mpnet-base-v2', response_dir='outputs/exp6_70b_full', cache_dir=None):
    print("retrieving and postprocessing responses...")
    model = SentenceTransformer(sbert_model_name, cache_folder=cache_dir)
    labels = []
    responses = []
    for file in tqdm(glob.glob(os.path.join(response_dir, '*.json'))):
        cur_labels, cur_responses = load_strategy_and_response(file)
        for response, label in zip(cur_responses, cur_labels):
            response = postprocess_response(response, label)
            responses.append(response)
            labels.append(label)

    print("encoding responses...")
    response_encodings = encode_responses(model, responses)
    output_path = os.path.join(response_dir, 'response_encodings.pkl')

    with open(output_path, 'wb') as f:
        pickle.dump((response_encodings, labels), f)


if __name__ == '__main__':
    fire.Fire(run)
