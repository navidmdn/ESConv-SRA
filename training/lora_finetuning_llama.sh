#
#WANDB_MODE=online WANDB_ENTITY=navidmdn WANDB_PROJECT=esconv-sra-lora python lora_finetuning_llama.py\
#  --train_file data/train_processed.json\
#  --dev_dir data/\
#  --output_dir outputs/llama2_lora_r128a256\
#  --do_train\
#  --do_eval\
#  --model_id meta-llama/Llama-2-7b-chat-hf\
#  --use_peft\
#  --cache_dir ../../hfcache\
#  --include_inputs_for_metrics\
#  --per_device_train_batch_size 2\
#  --per_device_eval_batch_size 4\
#  --gradient_accumulation_steps 32\
#  --num_train_epochs 5\
#  --save_strategy steps\
#  --eval_strategy steps\
#  --save_total_limit 1\
#  --metric_for_best_model "main_val_loss"\
#  --save_steps 20\
#  --logging_steps 1\
#  --overwrite_output_dir\
#  --report_to wandb\
#  --load_best_model_at_end\
#  --eval_steps 20\
#  --max_new_tokens 1024\
#  --warmup_ratio 0.1\
#  --lr_scheduler_type "cosine_with_restarts"\
#  --lr_scheduler_kwargs '{"num_cycles": 5}'\
#  --lora_r 128\
#  --lora_a 256


WANDB_MODE=online WANDB_ENTITY=navidmdn WANDB_PROJECT=esconv-sra-lora python lora_finetuning_llama.py\
  --train_file data/train_processed.json\
  --dev_dir data/\
  --output_dir outputs/llama3_lora_r128a256\
  --do_train\
  --do_eval\
  --model_id meta-llama/Llama-3.1-8B-Instruct\
  --use_peft\
  --cache_dir ../../hfcache\
  --include_inputs_for_metrics\
  --per_device_train_batch_size 2\
  --per_device_eval_batch_size 4\
  --gradient_accumulation_steps 32\
  --num_train_epochs 5\
  --save_strategy steps\
  --eval_strategy steps\
  --save_total_limit 1\
  --metric_for_best_model "main_val_loss"\
  --save_steps 20\
  --logging_steps 5\
  --overwrite_output_dir\
  --report_to wandb\
  --load_best_model_at_end\
  --eval_steps 20\
  --max_new_tokens 1024\
  --warmup_ratio 0.1\
  --lr_scheduler_type "cosine_with_restarts"\
  --lr_scheduler_kwargs '{"num_cycles": 5}'\
  --lora_r 128\
  --lora_a 256