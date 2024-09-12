python mt_evaluation.py\
 --model1_response_file data/llama2_ft.json\
 --model2_response_file data/llama2_standard.json\
 --evaluator_prompt_file strategy_following_comparison_judge_prompt.txt\
 --evaluator_llm_name gpt-4o

python mt_evaluation.py\
 --model1_response_file data/llama3_ft.json\
 --model2_response_file data/llama3_standard.json\
 --evaluator_prompt_file strategy_following_comparison_judge_prompt.txt\
 --evaluator_llm_name gpt-4o
