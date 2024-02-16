PYTHONPATH=.. python multiple_strategy_continuation.py --model_path meta-llama/Llama-2-70b-chat-hf\
  --output_path outputs/exp3_70b_c1_hf_partial --get_attentions True --prompt_constructor partial --n_iters 100\
  --min_turn 3 --max_turn 10 --n_turns_as_conv 1 --history_first True

PYTHONPATH=.. python multiple_strategy_continuation.py --model_path meta-llama/Llama-2-70b-chat-hf\
  --output_path outputs/exp3_70b_c3_hf_partial --get_attentions True --prompt_constructor partial --n_iters 100\
  --min_turn 3 --max_turn 10 --n_turns_as_conv 3 --history_first True

PYTHONPATH=.. python multiple_strategy_continuation.py --model_path meta-llama/Llama-2-70b-chat-hf\
  --output_path outputs/exp3_70b_c5_hf_partial --get_attentions True --prompt_constructor partial --n_iters 100\
  --min_turn 3 --max_turn 10 --n_turns_as_conv 5 --history_first True

PYTHONPATH=.. python multiple_strategy_continuation.py --model_path meta-llama/Llama-2-70b-chat-hf\
  --output_path outputs/exp3_70b_c1_hl_partial --get_attentions True --prompt_constructor partial --n_iters 100\
  --min_turn 3 --max_turn 10 --n_turns_as_conv 1 --history_first False

PYTHONPATH=.. python multiple_strategy_continuation.py --model_path meta-llama/Llama-2-70b-chat-hf\
  --output_path outputs/exp3_70b_c3_hl_partial --get_attentions True --prompt_constructor partial --n_iters 100\
  --min_turn 3 --max_turn 10 --n_turns_as_conv 3 --history_first False

PYTHONPATH=.. python multiple_strategy_continuation.py --model_path meta-llama/Llama-2-70b-chat-hf\
  --output_path outputs/exp3_70b_c5_hl_partial --get_attentions True --prompt_constructor partial --n_iters 100\
  --min_turn 3 --max_turn 10 --n_turns_as_conv 5 --history_first False

PYTHONPATH=.. python multiple_strategy_continuation.py --model_path meta-llama/Llama-2-70b-chat-hf\
  --output_path outputs/exp3_70b_full --get_attentions True --prompt_constructor full --n_iters 100\
  --min_turn 3 --max_turn 10
