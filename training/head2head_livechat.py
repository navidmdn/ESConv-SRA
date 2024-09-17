import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, LlamaForCausalLM, TextIteratorStreamer
from fire import Fire
from typing import List, Tuple
import torch
import json
from prompting.llama_prompt import modified_extes_support_strategies



class SysMsg:
    def __init__(self):
        self.cur_strategy = None

    def __str__(self):
        if self.cur_strategy is None:
            return ""

        description = modified_extes_support_strategies[self.cur_strategy]
        return """You are a helpful and caring AI which is an expert in emotional support.\
 A user has come to you for help. Use "{cur_strategy}" strategy ({strategy_description}) for answering the user.\
 make your response short and to the point.""".format(cur_strategy=self.cur_strategy,
                                                      strategy_description=description)


# todo: add streamer to generation
def run(model_name_or_path_1='Qwen/Qwen1.5-0.5B-Chat', model_name_or_path_2='meta.llama3-8b-instruct-v1:0',
        max_new_tokens=512, cache_dir=None,
        temperature=0.7, top_p=0.9, num_return_sequences=1, do_sample=True, num_beams=1, share=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_1 = AutoModelForCausalLM.from_pretrained(model_name_or_path_1, cache_dir=cache_dir)
    model_1 = model_1.to(device)
    model_1.eval()
    tokenizer_1 = AutoTokenizer.from_pretrained(model_name_or_path_1, cache_dir=cache_dir)

    model_2 = AutoModelForCausalLM.from_pretrained(model_name_or_path_2, cache_dir=cache_dir)
    model_2 = model_2.to(device)
    model_2.eval()
    tokenizer_2 = AutoTokenizer.from_pretrained(model_name_or_path_2, cache_dir=cache_dir)

    # Keep track of the selected strategy
    strategies = list(modified_extes_support_strategies.keys())
    cur_sys_msg = SysMsg()

    def set_strategy(strategy_str: str):
        """ Update the selected strategy based on user input """
        cur_sys_msg.cur_strategy = strategy_str

        print(f"Strategy set to: {cur_sys_msg.cur_strategy}")
        return f"Strategy set to: {cur_sys_msg.cur_strategy}"

    def clear_chat_and_reset_strategy():
        """ Clear chat and reset the strategy """
        cur_sys_msg.cur_strategy = None
        print("Chat and strategy cleared")
        return [], [], "No strategy selected"

    def build_input_from_chat_history(chat_history: List[Tuple], msg: str):
        """ Build the input by combining the chat history with the system message. """
        messages = [{'role': 'system', 'content': str(cur_sys_msg)}]
        for user_msg, ai_msg in chat_history:
            messages.append({'role': 'user', 'content': user_msg})
            messages.append({'role': 'assistant', 'content': ai_msg})
        messages.append({'role': 'user', 'content': msg})
        print(messages)
        return messages

    # Define the chat function for model 1
    def chat_model(model, tokenizer, message, chat_history):
        messages = build_input_from_chat_history(chat_history, message)
        input_ids = tokenizer_1.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True
        )
        input_ids = input_ids.to(device)
        output = model.generate(input_ids, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id,
                                do_sample=do_sample, temperature=temperature, top_p=top_p,
                                num_return_sequences=num_return_sequences, num_beams=num_beams)
        response_token_ids = output[0][input_ids.shape[1]:]
        response = tokenizer.decode(response_token_ids, skip_special_tokens=True)
        chat_history.append((message, response))
        return "", chat_history

    def m1_fn(msg, chat_history):
        return chat_model(model_1, tokenizer_1, msg, chat_history)

    def m2_fn(msg, chat_history):
        return chat_model(model_2, tokenizer_2, msg, chat_history)

    iface = gr.Blocks()

    with iface:
        gr.Markdown("### Choose a Strategy")
        strategy_textbox = gr.Textbox(value="No strategy selected", label="Current Strategy", interactive=False)
        with gr.Row():
            for strategy in strategies:
                s = gr.Button(strategy, elem_id=f"strategy_{strategy}")
                s.click(lambda s=strategy: set_strategy(s), None, strategy_textbox)


        gr.Markdown("# Compare Two Models Side by Side")
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Model 1")
                chatbox_1 = gr.Chatbot([], layout="bubble", bubble_full_width=False)

            with gr.Column():
                gr.Markdown("## Model 2")
                chatbox_2 = gr.Chatbot([], layout="bubble", bubble_full_width=False)

        with gr.Row():
            msg = gr.Textbox(label="Your message here")
            msg.submit(m1_fn, [msg, chatbox_1], [msg, chatbox_1])
            msg.submit(m2_fn, [msg, chatbox_2], [msg, chatbox_2])
            clear = gr.Button("Clear Chat and Strategy")
            clear.click(clear_chat_and_reset_strategy, None, [chatbox_1, chatbox_2, strategy_textbox], queue=False)


    # Launch the app
    iface.launch(share=share)


if __name__ == '__main__':
    Fire(run)
