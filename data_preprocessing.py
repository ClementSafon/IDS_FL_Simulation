import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import os
from tqdm import tqdm
import math
import json

from sklearn.preprocessing import MinMaxScaler


def split_data(data, n, random_state=None):
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(data))
    partition_size = len(data) // n
    data_parts = []
    for i in range(n):
        start = i * partition_size
        end = (i + 1) * partition_size if i < n - 1 else len(data)
        data_parts.append(data.iloc[shuffled_indices[start:end]])
    return data_parts



def preprocess_data(data):
    # Preprocess the data
    m_data = data["attack_cat"]
    data = data.drop(columns=["label", "attack_cat"])

    x_data = pd.get_dummies(data)

    scaler = MinMaxScaler()
    scaler.fit(x_data)
    x_data[x_data.columns] = scaler.transform(x_data)

    y_data = m_data.apply(lambda x: False if x == "Normal" else True)
    y_data = pd.get_dummies(y_data, prefix="Malicious")

    return x_data, y_data, m_data


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", help="number of party datasets to split into", type=int)
    parser.add_argument("-trf", help="filepath to training data")
    parser.add_argument("-tef", help="filepath to testing data")
    args = parser.parse_args()

    train_data = pd.read_csv(args.trf)
    test_data = pd.read_csv(args.tef)

    x_total_train, y_total_train, m_total_train = preprocess_data(train_data)
    x_total_test, y_total_test, m_total_test = preprocess_data(test_data)
    x_total_test = x_total_test.reindex(columns=x_total_train.columns, fill_value=0)
    y_total_test = y_total_test.reindex(columns=y_total_train.columns, fill_value=0)

    print("Shape :\n  - train : ", x_total_train.shape, "\n  - test : ", x_total_test.shape)

    seed = 1534

    x_train_parts = split_data(x_total_train, args.n, seed)
    y_train_parts = split_data(y_total_train, args.n, seed)
    x_test_parts = split_data(x_total_test, args.n, seed)
    y_test_parts = split_data(y_total_test, args.n, seed)
    m_train_parts = split_data(m_total_train, args.n, seed)
    m_test_parts = split_data(m_total_test, args.n, seed)

    print("\nExtracting and Preprocessing data for {} clients".format(args.n))
    for i in tqdm(range(args.n)):
        name = "data_party" + str(i)
        x_train = x_train_parts[i]
        x_test = x_test_parts[i]
        y_train = y_train_parts[i]
        y_test = y_test_parts[i]
        m_train = m_train_parts[i]
        m_test = m_test_parts[i]
        np.savez(name, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, m_train=m_train, m_test=m_test)

    data_party_features = {"n_features": x_total_train.shape[1]}
    with open("data_party_features.json", "w") as f:
        json.dump(data_party_features, f)
    
    print("All file saved")




