from datasets import load_dataset


dataset = load_dataset(
    'json',
    data_files={
        'train': 'datasets/processed_data/train.json',
        'dev': 'datasets/processed_data/dev.json',
        'test': 'datasets/processed_data/test.json'
    }
)

dataset.save_to_disk('datasets/nyt')
