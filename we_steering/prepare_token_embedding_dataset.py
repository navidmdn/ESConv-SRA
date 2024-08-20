import pickle
from tqdm import tqdm
import torch
import json
from copy import copy
import numpy as np
from transformers import LlamaTokenizer, LlamaModel
from typing import Dict, List

with open("../data/extended_esc_13b.json", "r") as f:
    data = json.load(f)

processed_convs = []

print("no of conversations: ", len(data))

for ex in data:
    messages = []
    speakers = ['user' if s == 'seeker' else 'assistant' for s in ex['speakers']]
    turns = [t.strip() for t in ex['dialog']]

    for s, t in zip(speakers, turns):
        messages.append({'role': s, 'content': t})

    for label, resp in ex['responses'].items():
        cur_msgs = copy(messages)
        cur_msgs.append({'role': 'assistant', 'content': resp})

        processed_convs.append({'messages': cur_msgs, 'strategy': label})

print("no of sampels: ", len(processed_convs))


np.random.shuffle(processed_convs)
sample = processed_convs[:1000]

tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", cache_dir='../../hfcache')
model = LlamaModel.from_pretrained("meta-llama/Llama-2-7b-chat-hf", cache_dir='../../hfcache')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
model.eval()


def get_assistant_response_embeddings(conv: List[Dict[str, str]]) -> np.ndarray:
    prompt_tokens = tokenizer.apply_chat_template(conv[:-1], tokenize=True)
    prompt_len = len(prompt_tokens)

    full_prompt = tokenizer.apply_chat_template(conv, tokenize=True)
    with torch.no_grad():
        outputs = model(input_ids=torch.tensor([full_prompt]).to(device))

    out_emb = outputs.last_hidden_state.cpu().numpy()[0]
    return out_emb[prompt_len:-1, :]


all_embs = []
labels = []

for ex in tqdm(sample):

    label = ex['strategy']
    conv = ex['messages']
    embs = get_assistant_response_embeddings(conv)

    labels.extend([label] * embs.shape[0])
    all_embs.append(embs)

all_embs = np.concatenate(all_embs, axis=0)
print("shape of embeddings: ", all_embs.shape)

with open('../data/sample_embeddings.pkl', 'wb') as f:
    pickle.dump((all_embs, labels), f)

