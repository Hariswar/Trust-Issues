import pandas as pd
from dataset import path_one, path_two
from pathlib import Path

path_one = Path(path_one)
path_two = Path(path_two)
data_format = ['title', 'text', 'label']
boolean = True
whitespace_check = r'\s+'
url_check = r'(http\S+|www\S+)'
character_check = r'[^a-zA-Z0-9\s]'
symbol_check = r'@[A-Za-z0-9_]+'
seed = 1

# get the data for both datasets. 
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

# cleaned the dataset and normalizing it. 
def clean_normalize_dataset(data):
    data = data.dropna()
    data = data.drop_duplicates(['text'])
    data = data.drop_duplicates(['title'])
    text_data = data['text'].str.lower()
    title_data = data['title'].str.lower()
    data['text'] = text_data
    data['title'] = title_data
    data['text'] = data['text'].str.strip()
    data['title'] = data['title'].str.strip()
    whitespace = data['text'].str.replace(whitespace_check, ' ', regex=boolean)
    whitespace_title = data['title'].str.replace(whitespace_check, ' ', regex=boolean)
    data['text'] = whitespace
    data['title'] = whitespace_title
    url = data['text'].str.replace(url_check, '', regex=boolean)
    url_title = data['title'].str.replace(url_check, '', regex=boolean)
    data['text'] = url
    data['title'] = url_title
    characters = data['text'].str.replace(character_check, '', regex=boolean)
    characters_title = data['title'].str.replace(character_check, '', regex=boolean)
    data['text'] = characters
    data['title'] = characters_title
    check = data['text'].str.replace(symbol_check, '', regex=boolean)
    check_title = data['title'].str.replace(symbol_check, '', regex=boolean)
    data['text'] = check
    data['title'] = check_title
    data = data.drop_duplicates(['text'])
    data = data.drop_duplicates(['title'])

    rows = len(data)
    data = data.sample(n=rows, random_state=seed) 
    return data

# created a functionality for split. Did 60% for training, 20% for testing and 20% for validation. 
def split_dataset(data):
    train = 0.6
    test = 0.2
    validation = 0.2
    rows = len(data)
    validation_split = int(validation * rows)
    train_split = int(train * rows)
    test_split = int(test * rows)
    train_data = data[:train_split]
    test_train = train_split + test_split 
    combine = train_split + test_split + validation_split
    test_data = data[train_split:test_train]
    validation_data = data[test_train:combine]
    return train_data, test_data, validation_data

# after cleaning and normalizing for dataset 1
evaluation = clean_normalize_dataset(read_evaluation)
train = clean_normalize_dataset(read_train)
test = clean_normalize_dataset(read_test)

# after cleaning, normalizing and splitting for dataset 2
dataset_two = clean_normalize_dataset(read_dataset_two)
dataset_two_train, dataset_two_test, dataset_two_validation = split_dataset(dataset_two)

# dataset 1
print(evaluation) 
print(train) 
print(test)

# datset 2 
print(dataset_two_validation) 
print(dataset_two_train) 
print(dataset_two_test)