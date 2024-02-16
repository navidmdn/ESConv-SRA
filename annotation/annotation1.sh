python annotation/strategy_following_comparison/prepare_data.py --mode prepare_comparison_data\
 --model1_output_dir prompting/outputs/exp6_70b_full/ --model2_output_dir prompting/outputs/exp6_70b_c1_hf_partial/ \
 --output_file annotation/strategy_following_comparison/data/70b_standard_vs_c1hf.json

python annotation/strategy_following_comparison/prepare_data.py --mode prepare_comparison_data\
 --model1_output_dir prompting/outputs/exp6_70b_full/ --model2_output_dir prompting/outputs/exp6_70b_c3_hf_partial/ \
 --output_file annotation/strategy_following_comparison/data/70b_standard_vs_c3hf.json

python annotation/strategy_following_comparison/prepare_data.py --mode prepare_comparison_data\
 --model1_output_dir prompting/outputs/exp6_70b_c3_hf_partial/ --model2_output_dir prompting/outputs/exp6_70b_c1_hf_partial/ \
 --output_file annotation/strategy_following_comparison/data/70b_c3hf_vs_c1hf.json