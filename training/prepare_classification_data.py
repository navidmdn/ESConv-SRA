import fire
import os
import random
import shutil
import json


def prepare_classification_data(
    input_dir: str = "data/splits/",
    output_dir: str = "data/post_processed/"
):

    for split in ['train', 'test', 'val']:
        with open(os.path.join(input_dir, f"{split}.json"), 'r') as f:
            data = json.load(f)
        strategy_cnt = {}
        dataset = []
        for example in data:
            for strategy, response in example['responses'].items():
                strategy_cnt[strategy] = strategy_cnt.get(strategy, 0) + 1
                dataset.append({
                    'sentence': response,
                    'strategy': strategy,
                })

        print(f"Number of examples: {len(dataset)}")
        print("total strategy classes: ", len(strategy_cnt))
        print(f"Strategy counts: {strategy_cnt}")

        random.shuffle(dataset)

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"{split}.json"), 'w') as f:
            for example in dataset:
                f.write(json.dumps(example) + '\n')


if __name__ == "__main__":
    fire.Fire(prepare_classification_data)