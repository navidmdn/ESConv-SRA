import torch
from typing import List

def default_aggregate_attention(attn):
    '''Extract average attention vector'''
    avged = []
    for layer in attn:
        layer_attns = layer.detach().cpu().squeeze(0)
        attns_per_head = layer_attns.mean(dim=0)
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


def heterogenous_stack(vecs):
    '''Pad vectors with zeros then stack'''
    max_length = max(v.shape[0] for v in vecs)
    return torch.stack([
        torch.concat((v, torch.zeros(max_length - v.shape[0])))
        for v in vecs
    ])


def get_tokens_str_list(tokenizer, tokens) -> List[str]:
    '''Turn tokens into text with mapping index'''
    chunks = tokenizer.convert_ids_to_tokens(tokens)

    new_tokens = []
    for token in chunks:
        # 9601 is the special token for split tokens in llama
        if token == '<0x0A>':
            new_tokens.append('\n')
        elif token.startswith(chr(9601)):
            new_tokens.append(f" {token[1:]}")
        else:
            new_tokens.append(token)
    return new_tokens


def build_attention_matrix(attentions, tokens, aggregate_fn=default_aggregate_attention):
    attn_m = heterogenous_stack([
        torch.tensor([
            1 if i == j else 0
            for j, token in enumerate(tokens)
        ], device=tokens.device)
        for i, token in enumerate(tokens)
    ] + list(map(aggregate_fn, attentions)))
    return attn_m

def build_attention_matrix_from_token_attn_pairs(tokens, attentions):
    attn_m = heterogenous_stack([
        torch.tensor([
            1 if i == j else 0
            for j, token in enumerate(tokens)
        ], device=tokens.device)
        for i, token in enumerate(tokens)
    ] + attentions)
    return attn_m

def get_completion(prompt, tokenizer, model):
    '''Get full text, token mapping, and attention matrix for a completion'''
    tokens = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)['input_ids'].to(model.device)
    outputs = model.generate(
        tokens,
        max_new_tokens=50,
        output_attentions=True,
        return_dict_in_generate=True,
        early_stopping=True,
        length_penalty=-1
    )

    sequences = outputs.sequences
    tokenized = get_tokens_str_list(tokenizer, sequences[0])
    decoded = tokenizer.decode(sequences[0], skip_special_tokens=False)

    attn_m = build_attention_matrix(outputs.attentions, tokens[0].detach().cpu())
    assert len(tokenized) == len(attn_m)
    return decoded, tokenized, attn_m



def show_matrix(xs):
    for x in xs:
        line = ''
        for y in x:
            line += '{:.4f}\t'.format(float(y))
        print(line)


