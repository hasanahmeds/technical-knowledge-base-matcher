import enum

import pandas as pd

from data import dataset


class ColumnType(enum.Enum):
    sentence1 = 0,
    sentence2 = 1,
    labels = 2,


columns = [
    ColumnType.sentence1.name,
    ColumnType.sentence2.name,
    ColumnType.labels.name,
]


class SNLIDataset(dataset.DatasetExperiment):
    
    def __init__(self, *args):
        super().__init__(*args)
        dataset = pd.read_csv(
            '{}{}'.format(self.data_dir, 'train_snli.txt'),
            delimiter='\t',
            header=None,
            names=columns,
            na_values='',
        )
        dataset.dropna(inplace=True)
        dataset = dataset.sample(frac=1, random_state=1).reset_index(drop=True)
        num_instances = len(dataset)
        self.num_train = num_instances * (1 - self.dev_ratio - self.test_ratio)
        self.num_dev = num_instances * self.dev_ratio
        self.num_test = num_instances * self.test_ratio
        self.train = dataset.loc[:self.num_train]
        self.dev = dataset.loc[self.num_train:self.num_train + self.num_dev]
        self.test = dataset.loc[
