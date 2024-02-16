# ESConv-SRA

This repository contains the code and the data for the paper "Steering Conversational Large Language Models for Long
Emotional Support Conversations"

## Data

The original ESConv dataset is available under `esconv/` directory. You can run the `process_esconv.sh` to
convert the data into a format that we use in our experiments. It will create a json file inside the same folder
called `conversations.json`. You can run the script with the following command:

```sh
bash process_esconv.sh
```

## Experiments

For our experiments we use LLaMa v2 chat models with 4bit quantization. You can follow the instruction in the following
links to get access to [7b](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), [13b](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) 
and [70b](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) models on huggingface.

All of the experiments are conducted using the `transformers` library. We use bitsandbytes to quantize the models. We also
run inference on the models using one A100 GPU with 80GB memory.

You can run the experiments in the paper using the following commands:

```sh
cd prompting
bash llama7b.sh
bash llama13b.sh
bash llama70b.sh
```

This will generate the sampled data collections for the experiments in the paper. The rest of the 
analysis will be done using `prompting/strategy_following_comparison.ipynb` notebook.
