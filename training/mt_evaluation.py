import os
import re
import fire
import json
from typing import List, Dict
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from tqdm import tqdm
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_community.callbacks import get_openai_callback


def build_conv_str(history_messages):
    conv_str = ""

    for msg in history_messages:
        conv_str += f"### {msg['role']}:\n{msg['content']}\n\n"

    return conv_str

def process_response(response: str) -> str:
    judge = re.findall("JUDGE: \[\[([ABC])\]\]", response)
    if judge:
        return judge[0]
    else:
        return

def run(model1_response_file: str, model2_response_file: str, evaluator_prompt_file: str = "strategy_following_comparison.txt",
        response_col: str = 'model_response', evaluator_llm_name: str = 'gpt-3.5-turbo',
        result_dir: str = 'data/', max_examples: int = -1):

    llm = ChatOpenAI(
        model_name=evaluator_llm_name,
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=OPENAI_API_BASE,
        max_tokens=1024,
        temperature=0.1,
        default_headers={"providers": "openai"}
    )

    with open(evaluator_prompt_file, 'r') as f:
        prompt_template_string = f.read().strip()

    prompt_template = ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate.from_template(prompt_template_string)
    ])

    chain = prompt_template | llm | StrOutputParser()

    with open(model1_response_file, 'r') as f:
        model1_responses = []
        for line in f:
            ex = json.loads(line)
            model1_responses.append(ex)

    with open(model2_response_file, 'r') as f:
        model2_responses = []
        for line in f:
            ex = json.loads(line)
            model2_responses.append(ex)

    if max_examples > 0:
        model1_responses = model1_responses[:max_examples]
        model2_responses = model2_responses[:max_examples]

    assert len(model1_responses) == len(model2_responses)

    model_1_name = model1_response_file.split('/')[-1].replace('.json', '')
    model_2_name = model2_response_file.split('/')[-1].replace('.json', '')

    result_file_name = f"{result_dir}/{model_1_name}_vs_{model_2_name}_fullconv_mtbench_results.json"

    results = []
    for ex1, ex2 in tqdm(zip(model1_responses, model2_responses)):
        # todo: assign a session id
        # assert ex1['session_id'] == ex2['session_id']

        assert len(ex1['messages']) == len(ex2['messages'])
        # assert ex1[query_col] == ex2[query_col]

        # last message is human answer and first is system message
        conv_history = ex1['messages'][1:-1]

        conv_history_str = build_conv_str(conv_history)

        res1 = chain.invoke({
            'strategy_instructions': ex1['messages'][0]['content'],
            'conversation_history': conv_history_str,
            'assistant_a_resp': ex1[response_col],
            'assistant_b_resp': ex2[response_col],
        })

        res2 = chain.invoke({
            'strategy_instructions': ex1['messages'][0]['content'],
            'conversation_history': conv_history_str,
            'assistant_a_resp': ex2[response_col],
            'assistant_b_resp': ex1[response_col],
        })

        judge1 = process_response(res1)
        judge2 = process_response(res2)

        if judge1 is None or judge2 is None:
            print("The judge response was not well structured. skipping this example")
            continue

        if judge1 == 'A' and judge2 == 'B':
            winner_model_name = model_1_name
        elif judge1 == 'B' and judge2 == 'A':
            winner_model_name = model_2_name
        else:
            winner_model_name = 'Tie'

        results.append({
            'history': conv_history,
            'model_a': model_1_name,
            'model_b': model_2_name,
            'response_a': ex1[response_col],
            'response_b': ex2[response_col],
            'judge_a': res1,
            'judge_b': res2,
            'winner': winner_model_name
        })

        with open(result_file_name, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')


if __name__ == '__main__':
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    OPENAI_API_BASE = os.environ.get('OPENAI_API_BASE')
    fire.Fire(run)

