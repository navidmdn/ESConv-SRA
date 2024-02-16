from flask import Flask, render_template
from attention_visualizer.attention import build_attention_matrix_from_token_attn_pairs,\
    get_tokens_str_list
import json
from transformers import AutoTokenizer
from glob import glob
import pickle
import os
import argparse


app = Flask(__name__)
app.config['DEBUG'] = True


def get_attention_matrix_and_tokens(example, tokenizer):
    tokens, attentions = example[list(example.keys())[0]]
    input_tokens = tokens[:len(tokens)-len(attentions)]
    attn_m = build_attention_matrix_from_token_attn_pairs(input_tokens, attentions)
    tokens = get_tokens_str_list(tokenizer, tokens)
    sparse = attn_m.to_sparse()

    return sparse, tokens

def get_outputs_from_file(dir_name):
    attention_files = glob(os.path.join(dir_name, "*_attentions.pkl"))
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-70b-chat-hf')
    examples = []
    for attention_file in attention_files:
        with open(attention_file, 'rb') as f:
            examples.append(pickle.load(f))

    return examples, tokenizer


@app.route('/attention/<int:i>', methods=['GET'])
def attention_view(i):
    i = i % len(examples)
    sparse, tokenized = get_attention_matrix_and_tokens(examples[i], tokenizer)

    indices, values = sparse.indices(), sparse.values()
    return json.dumps({
        'tokens': tokenized,
        'attn_indices': indices.T.numpy().tolist(),
        'attn_values': values.numpy().tolist(),
    })


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='The path to data the folder')
    args = parser.parse_args()
    print(args.data_dir)
    examples, tokenizer = get_outputs_from_file(args.data_dir)

    app.run(debug=True, port=5220, host='localhost')


