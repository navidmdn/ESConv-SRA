PYTHONPATH=.. python multiple_strategy_continuation.py --model_path meta-llama/Llama-2-7b-chat-hf\
  --cache_dir ../../hfcache/ --output_path outputs/extended_esc_7b --get_attentions False --prompt_constructor partial --n_iters -1\
  --min_turn 3 --max_turn 12 --n_turns_as_conv 1 --history_first True

PYTHONPATH=.. python multiple_strategy_continuation.py --model_path meta-llama/Llama-2-13b-chat-hf\
  --cache_dir ../../hfcache/ --output_path outputs/extended_esc_13b --get_attentions False --prompt_constructor partial --n_iters -1\
  --min_turn 3 --max_turn 12 --n_turns_as_conv 1 --history_first True

PYTHONPATH=.. python multiple_strategy_continuation.py --model_path meta-llama/Llama-2-70b-chat-hf\
  --cache_dir ../../hfcache/ --output_path outputs/extended_esc_70b--get_attentions False --prompt_constructor partial --n_iters -1\
  --min_turn 3 --max_turn 10 --n_turns_as_conv 1 --history_first True
