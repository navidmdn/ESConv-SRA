PYTHONPATH=.. python multiple_strategy_continuation.py --model_path mistralai/Mistral-7B-Instruct-v0.2\
  --output_path outputs/mistral7b_c1_hf_partial --get_attentions True --prompt_constructor partial --n_iters 25\
  --min_turn 3 --max_turn 12 --n_turns_as_conv 1 --history_first True --cache_dir ../../hfcache/

PYTHONPATH=.. python multiple_strategy_continuation.py --model_path mistralai/Mistral-7B-Instruct-v0.2\
  --output_path outputs/mistral7b_c3_hf_partial --get_attentions True --prompt_constructor partial --n_iters 25\
  --min_turn 3 --max_turn 12 --n_turns_as_conv 3 --history_first True --cache_dir ../../hfcache/

PYTHONPATH=.. python multiple_strategy_continuation.py --model_path mistralai/Mistral-7B-Instruct-v0.2\
  --output_path outputs/mistral7b_c5_hf_partial --get_attentions True --prompt_constructor partial --n_iters 25\
  --min_turn 3 --max_turn 12 --n_turns_as_conv 5 --history_first True --cache_dir ../../hfcache/

PYTHONPATH=.. python multiple_strategy_continuation.py --model_path mistralai/Mistral-7B-Instruct-v0.2\
  --output_path outputs/mistral7b_c1_hl_partial --get_attentions True --prompt_constructor partial --n_iters 25\
  --min_turn 3 --max_turn 12 --n_turns_as_conv 1 --history_first False --cache_dir ../../hfcache/

PYTHONPATH=.. python multiple_strategy_continuation.py --model_path mistralai/Mistral-7B-Instruct-v0.2\
  --output_path outputs/mistral7b_c3_hl_partial --get_attentions True --prompt_constructor partial --n_iters 25\
  --min_turn 3 --max_turn 12 --n_turns_as_conv 3 --history_first False --cache_dir ../../hfcache/

PYTHONPATH=.. python multiple_strategy_continuation.py --model_path mistralai/Mistral-7B-Instruct-v0.2\
  --output_path outputs/mistral7b_c5_hl_partial --get_attentions True --prompt_constructor partial --n_iters 25\
  --min_turn 3 --max_turn 12 --n_turns_as_conv 5 --history_first False --cache_dir ../../hfcache/

PYTHONPATH=.. python multiple_strategy_continuation.py --model_path mistralai/Mistral-7B-Instruct-v0.2\
  --output_path outputs/mistral7b_full --get_attentions True --prompt_constructor full --n_iters 25\
  --min_turn 3 --max_turn 12 --cache_dir ../../hfcache/
