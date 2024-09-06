# llama2 responses

#python generate_baseline_responses.py\
# --test_path data/test_processed.json\
# --model_path outputs/llama2_lora_r128a256\
# -save_path data/eval4/llama2_ft.json\
# --baseline_name "standard"\
# --base_model_type llama2\
# --max_test_samples 1000\
# --cache_dir ../../hfcache\
# --infere_strategy\
# --strategy_classifer_path outputs/roberta_large_strategy_classifier\
# --save_batch_size 1\
# --get_sra
#
#python generate_baseline_responses.py\
# --test_path data/test_processed.json\
# --model_path meta-llama/Llama-2-7b-chat-hf\
# -save_path data/eval4/llama2_c1hf.json\
# --baseline_name "c1_hf"\
# --base_model_type llama2\
# --max_test_samples 1000\
# --cache_dir ../../hfcache\
# --infere_strategy\
# --save_batch_size 1\
# --strategy_classifer_path outputs/roberta_large_strategy_classifier\
# --get_sra
#
#
#python generate_baseline_responses.py\
# --test_path data/test_processed.json\
# --model_path meta-llama/Llama-2-7b-chat-hf\
# -save_path data/eval4/llama2_c3hf.json\
# --baseline_name "c3_hf"\
# --base_model_type llama2\
# --max_test_samples 1000\
# --cache_dir ../../hfcache\
# --infere_strategy\
# --save_batch_size 1\
# --strategy_classifer_path outputs/roberta_large_strategy_classifier\
# --get_sra
#
#
#python generate_baseline_responses.py\
# --test_path data/test_processed.json\
# --model_path meta-llama/Llama-2-7b-chat-hf\
# -save_path data/eval4/llama2_c5hf.json\
# --baseline_name "c5_hf"\
# --base_model_type llama2\
# --max_test_samples 1000\
# --cache_dir ../../hfcache\
# --infere_strategy\
# --save_batch_size 1\
# --strategy_classifer_path outputs/roberta_large_strategy_classifier\
# --get_sra
#
#
#python generate_baseline_responses.py\
# --test_path data/test_processed.json\
# --model_path meta-llama/Llama-2-7b-chat-hf\
# -save_path data/eval4/llama2_c1hl.json\
# --baseline_name "c1_hl"\
# --base_model_type llama2\
# --max_test_samples 1000\
# --cache_dir ../../hfcache\
# --infere_strategy\
# --save_batch_size 1\
# --strategy_classifer_path outputs/roberta_large_strategy_classifier\
# --get_sra
#
#
#python generate_baseline_responses.py\
# --test_path data/test_processed.json\
# --model_path meta-llama/Llama-2-7b-chat-hf\
# -save_path data/eval4/llama2_c3hl.json\
# --baseline_name "c3_hl"\
# --base_model_type llama2\
# --max_test_samples 1000\
# --cache_dir ../../hfcache\
# --save_batch_size 1\
# --infere_strategy\
# --strategy_classifer_path outputs/roberta_large_strategy_classifier\
# --get_sra
#
#
#python generate_baseline_responses.py\
# --test_path data/test_processed.json\
# --model_path meta-llama/Llama-2-7b-chat-hf\
# -save_path data/eval4/llama2_c5hl.json\
# --baseline_name "c5_hl"\
# --base_model_type llama2\
# --max_test_samples 1000\
# --cache_dir ../../hfcache\
# --infere_strategy\
# --save_batch_size 1\
# --strategy_classifer_path outputs/roberta_large_strategy_classifier\
# --get_sra
#
#
#python generate_baseline_responses.py\
# --test_path data/test_processed.json\
# --model_path meta-llama/Llama-2-7b-chat-hf\
# -save_path data/eval4/llama2_standard.json\
# --baseline_name "standard"\
# --base_model_type llama2\
# --max_test_samples 1000\
# --cache_dir ../../hfcache\
# --infere_strategy\
# --save_batch_size 1\
# --strategy_classifer_path outputs/roberta_large_strategy_classifier\
# --get_sra

# llama3 responses

python generate_baseline_responses.py\
 --test_path data/test_processed.json\
 --model_path outputs/llama3_lora_r128a256/checkpoint-760\
 -save_path data/eval4/llama3_ft.json\
 --baseline_name "standard"\
 --base_model_type llama3\
 --max_test_samples 1000\
 --cache_dir ../../hfcache\
 --infere_strategy\
 --strategy_classifer_path outputs/roberta_large_strategy_classifier\
 --save_batch_size 1\
 --get_sra

python generate_baseline_responses.py\
 --test_path data/test_processed.json\
 --model_path meta-llama/Meta-Llama-3-8B-Instruct\
 -save_path data/eval4/llama3_c1hf.json\
 --baseline_name "c1_hf"\
 --base_model_type llama3\
 --max_test_samples 1000\
 --cache_dir ../../hfcache\
 --infere_strategy\
 --save_batch_size 1\
 --strategy_classifer_path outputs/roberta_large_strategy_classifier\
 --get_sra


python generate_baseline_responses.py\
 --test_path data/test_processed.json\
 --model_path meta-llama/Meta-Llama-3-8B-Instruct\
 -save_path data/eval4/llama3_c3hf.json\
 --baseline_name "c3_hf"\
 --base_model_type llama3\
 --max_test_samples 1000\
 --cache_dir ../../hfcache\
 --infere_strategy\
 --save_batch_size 1\
 --strategy_classifer_path outputs/roberta_large_strategy_classifier\
 --get_sra


python generate_baseline_responses.py\
 --test_path data/test_processed.json\
 --model_path meta-llama/Meta-Llama-3-8B-Instruct\
 -save_path data/eval4/llama3_c5hf.json\
 --baseline_name "c5_hf"\
 --base_model_type llama3\
 --max_test_samples 1000\
 --cache_dir ../../hfcache\
 --infere_strategy\
 --save_batch_size 1\
 --strategy_classifer_path outputs/roberta_large_strategy_classifier\
 --get_sra


python generate_baseline_responses.py\
 --test_path data/test_processed.json\
 --model_path meta-llama/Meta-Llama-3-8B-Instruct\
 -save_path data/eval4/llama3_c1hl.json\
 --baseline_name "c1_hl"\
 --base_model_type llama3\
 --max_test_samples 1000\
 --cache_dir ../../hfcache\
 --infere_strategy\
 --save_batch_size 1\
 --strategy_classifer_path outputs/roberta_large_strategy_classifier\
 --get_sra


python generate_baseline_responses.py\
 --test_path data/test_processed.json\
 --model_path meta-llama/Meta-Llama-3-8B-Instruct\
 -save_path data/eval4/llama3_c3hl.json\
 --baseline_name "c3_hl"\
 --base_model_type llama3\
 --max_test_samples 1000\
 --cache_dir ../../hfcache\
 --save_batch_size 1\
 --infere_strategy\
 --strategy_classifer_path outputs/roberta_large_strategy_classifier\
 --get_sra


python generate_baseline_responses.py\
 --test_path data/test_processed.json\
 --model_path meta-llama/Meta-Llama-3-8B-Instruct\
 -save_path data/eval4/llama3_c5hl.json\
 --baseline_name "c5_hl"\
 --base_model_type llama3\
 --max_test_samples 1000\
 --cache_dir ../../hfcache\
 --infere_strategy\
 --save_batch_size 1\
 --strategy_classifer_path outputs/roberta_large_strategy_classifier\
 --get_sra


python generate_baseline_responses.py\
 --test_path data/test_processed.json\
 --model_path meta-llama/Meta-Llama-3-8B-Instruct\
 -save_path data/eval4/llama3_standard.json\
 --baseline_name "standard"\
 --base_model_type llama3\
 --max_test_samples 1000\
 --cache_dir ../../hfcache\
 --infere_strategy\
 --save_batch_size 1\
 --strategy_classifer_path outputs/roberta_large_strategy_classifier\
 --get_sra

