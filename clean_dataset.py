import pandas as pd
from dataset import path_one, path_two
from pathlib import Path

path_one = Path(path_one)
path_two = Path(path_two)
data_format = ['title', 'text', 'label']

read_evaluation = pd.read_csv(path_one / "evaluation.csv", sep=';')
drop_columns = read_evaluation.columns[0]
read_evaluation = read_evaluation.drop(drop_columns, axis=1)
read_test = pd.read_csv(path_one / "test (1).csv", sep=';')
drop_columns_test = read_test.columns[0]
read_test = read_test.drop(drop_columns_test, axis=1)
read_train = pd.read_csv(path_one / "train (2).csv", sep=';')
drop_columns_train = read_train.columns[0]
read_train = read_train.drop(drop_columns_train, axis=1)

read_dataset_two = pd.read_csv(path_two / "WELFake_Dataset.csv", sep=',')
drop_columns_two = read_dataset_two.columns[0]
read_dataset_two = read_dataset_two.drop(drop_columns_two, axis=1)

print(read_evaluation)
print(read_test)
print(read_train)
print(read_dataset_two)


