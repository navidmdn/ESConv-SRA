python strategy_following_comparison/prepare_data.py --mode prepare_annotation_data\
 --comparison_data_file strategy_following_comparison/data/70b_standard_vs_c1hf.json\
 --output_file strategy_following_comparison/data/70b_standard_vs_c1hf_annotations.json\
 --max_samples 45

python strategy_following_comparison/prepare_data.py --mode prepare_annotation_data\
 --comparison_data_file strategy_following_comparison/data/70b_standard_vs_c3hf.json\
 --output_file strategy_following_comparison/data/70b_standard_vs_c3hf_annotations.json\
 --max_samples 45

python strategy_following_comparison/prepare_data.py --mode prepare_annotation_data\
 --comparison_data_file strategy_following_comparison/data/70b_c3hf_vs_c1hf.json\
 --output_file strategy_following_comparison/data/70b_c3hf_vs_c1hf_annotations.json\
 --max_samples 45