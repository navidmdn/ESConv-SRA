import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, LlamaForCausalLM, TextIteratorStreamer
from fire import Fire
from typing import List, Tuple
import torch
from transformers import BitsAndBytesConfig
from langchain_aws import ChatBedrock



# todo: add streamer to generation
def run(model_name_or_path_1='Qwen/Qwen1.5-0.5B-Chat', model_name_or_path_2='meta.llama3-8b-instruct-v1:0',
        max_new_tokens=512,
        temperature=0.7, top_p=0.9, num_return_sequences=1, do_sample=True, num_beams=1, share=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model_1 = AutoModelForCausalLM.from_pretrained(model_name_or_path_1)
    model_1 = model_1.to(device)
    model_1.eval()
    tokenizer_1 = AutoTokenizer.from_pretrained(model_name_or_path_1)

    model_2 = AutoModelForCausalLM.from_pretrained(model_name_or_path_2)
    model_2 = model_2.to(device)
    model_2.eval()
    tokenizer_2 = AutoTokenizer.from_pretrained(model_name_or_path_2)


    def build_input_from_chat_history(chat_history: List[Tuple], msg: str):
        messages = [{'role': 'system', 'content': ""}]
        for user_msg, ai_msg in chat_history:
            messages.append({'role': 'user', 'content': user_msg})
            messages.append({'role': 'assistant', 'content': ai_msg})
        messages.append({'role': 'user', 'content': msg})
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


    iface = gr.Blocks()

    with iface:
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
            msg.submit(chat_model, [model_1, tokenizer_1, msg, chatbox_1], [msg, chatbox_1])
            msg.submit(chat_model, [model_2, tokenizer_2, msg, chatbox_2], [msg, chatbox_2])
            clear = gr.Button("Clear Chat")
            clear.click(lambda: None, None, chatbox_1, queue=False)
            clear.click(lambda: None, None, chatbox_2, queue=False)

    # Launch the app
    iface.launch(share=share)


if __name__ == '__main__':
    Fire(run)