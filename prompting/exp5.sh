PYTHONPATH=.. python multiple_strategy_continuation.py --model_path meta-llama/Llama-2-13b-chat-hf\
  --cache_dir ../../hfcache/ --output_path outputs/exp5_13b_c1_hf_partial --get_attentions True --prompt_constructor partial --n_iters 300\
  --min_turn 3 --max_turn 12 --n_turns_as_conv 1 --history_first True

PYTHONPATH=.. python multiple_strategy_continuation.py --model_path meta-llama/Llama-2-13b-chat-hf\
  --cache_dir ../../hfcache/ --output_path outputs/exp5_13b_c3_hf_partial --get_attentions True --prompt_constructor partial --n_iters 300\
  --min_turn 3 --max_turn 12 --n_turns_as_conv 3 --history_first True

PYTHONPATH=.. python multiple_strategy_continuation.py --model_path meta-llama/Llama-2-13b-chat-hf\
  --cache_dir ../../hfcache/ --output_path outputs/exp5_13b_c5_hf_partial --get_attentions True --prompt_constructor partial --n_iters 300\
  --min_turn 3 --max_turn 12 --n_turns_as_conv 5 --history_first True

PYTHONPATH=.. python multiple_strategy_continuation.py --model_path meta-llama/Llama-2-13b-chat-hf\
  --cache_dir ../../hfcache/ --output_path outputs/exp5_13b_c1_hl_partial --get_attentions True --prompt_constructor partial --n_iters 300\
  --min_turn 3 --max_turn 12 --n_turns_as_conv 1 --history_first False

PYTHONPATH=.. python multiple_strategy_continuation.py --model_path meta-llama/Llama-2-13b-chat-hf\
  --cache_dir ../../hfcache/ --output_path outputs/exp5_13b_c3_hl_partial --get_attentions True --prompt_constructor partial --n_iters 300\
  --min_turn 3 --max_turn 12 --n_turns_as_conv 3 --history_first False

PYTHONPATH=.. python multiple_strategy_continuation.py --model_path meta-llama/Llama-2-13b-chat-hf\
  --cache_dir ../../hfcache/ --output_path outputs/exp5_13b_c5_hl_partial --get_attentions True --prompt_constructor partial --n_iters 300\
  --min_turn 3 --max_turn 12 --n_turns_as_conv 5 --history_first False

PYTHONPATH=.. python multiple_strategy_continuation.py --model_path meta-llama/Llama-2-13b-chat-hf\
  --cache_dir ../../hfcache/ --output_path outputs/exp5_13b_full --get_attentions True --prompt_constructor full --n_iters 300\
  --min_turn 3 --max_turn 12
