import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import os
from tqdm import tqdm
import math
import json

from sklearn.preprocessing import MinMaxScaler


def split_data_random(data, n, random_state=None):
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(data))
    partition_size = len(data) // n
    data_parts = []
    for i in range(n):
        start = i * partition_size
        end = (i + 1) * partition_size if i < n - 1 else len(data)
        data_parts.append(data.iloc[shuffled_indices[start:end]])
    return data_parts

def split_data(data, sorted_index_lists):
    data_parts = []
    for sorted_index_list in sorted_index_lists:
        data_parts.append(data.iloc[sorted_index_list])
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
    parser.add_argument("-f", help="folder to save data to")
    args = parser.parse_args()

    folder_name = "data_client_" + args.f

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    train_data = pd.read_csv(args.trf)
    test_data = pd.read_csv(args.tef)

    x_total_train, y_total_train, m_total_train = preprocess_data(train_data)
    x_total_test, y_total_test, m_total_test = preprocess_data(test_data)
    x_total_test = x_total_test.reindex(columns=x_total_train.columns, fill_value=0)
    y_total_test = y_total_test.reindex(columns=y_total_train.columns, fill_value=0)

    print("Shape :\n  - train : ", x_total_train.shape, "\n  - test : ", x_total_test.shape)

    # split data randomly
    # seed = 1534
    # x_train_parts = split_data_random(x_total_train, args.n, seed)
    # y_train_parts = split_data_random(y_total_train, args.n, seed)
    # x_test_parts = split_data_random(x_total_test, args.n, seed)
    # y_test_parts = split_data_random(y_total_test, args.n, seed)
    # m_train_parts = split_data_random(m_total_train, args.n, seed)
    # m_test_parts = split_data_random(m_total_test, args.n, seed)

    # split data by attack category
    clients_special_distribution = {
            "Normal": None,
            "Fuzzers": None,
            "Analysis": None,
            "Backdoor": [0, 0, 1],
            "DoS": None,
            "Exploits": None,
            "Generic": None,
            "Reconnaissance": None,
            "Shellcode": None,
            "Worms": None
        }
    seed=13458
    
    attack_cat_index = {}
    for i, attack_cat in enumerate(clients_special_distribution.keys()):
        attack_cat_index[attack_cat] = np.array(m_total_train[m_total_train == attack_cat].index.tolist())

    sorted_index_lists = [[] for _ in range(args.n)]

    for attack_cat in clients_special_distribution.keys():
        if clients_special_distribution[attack_cat] is None:
            np.random.seed(seed)
            shuffled_indices = np.random.permutation(len(attack_cat_index[attack_cat]))
            partition_size = math.ceil(len(attack_cat_index[attack_cat]) / args.n)
            for i in range(args.n):
                start = i * partition_size
                end = (i + 1) * partition_size if i < args.n - 1 else len(attack_cat_index[attack_cat])
                sorted_index_lists[i].extend(attack_cat_index[attack_cat][shuffled_indices[start:end]])
        else:
            partition_sizes = [
                math.ceil(len(attack_cat_index[attack_cat]) * val)
                for val in clients_special_distribution[attack_cat]
            ]
            np.random.seed(seed)
            np.random.shuffle(attack_cat_index[attack_cat])
            for i in range(args.n):
                start = sum(partition_sizes[:i])
                end = sum(partition_sizes[:i + 1])
                sorted_index_lists[i].extend(attack_cat_index[attack_cat][start:end])
    
    for i in range(args.n):
        sorted_index_lists[i] = np.array(sorted_index_lists[i])
    
    # check if all the data is taken
    total_index = np.concatenate(sorted_index_lists)
    try:
        assert len(np.unique(total_index)) == len(x_total_train)
    except AssertionError as e:
        print("WARNING : some data are not taken into account")


    x_train_parts = split_data(x_total_train, sorted_index_lists)
    y_train_parts = split_data(y_total_train, sorted_index_lists)
    x_test_parts = split_data_random(x_total_test, args.n, seed)
    y_test_parts = split_data_random(y_total_test, args.n, seed)
    m_train_parts = split_data_random(m_total_train, args.n, seed)
    m_test_parts = split_data_random(m_total_test, args.n, seed)
    


    print("\nExtracting and Preprocessing data for {} clients".format(args.n))
    for i in tqdm(range(args.n)):
        name = folder_name + "/party" + str(i)
        x_train = x_train_parts[i]
        x_test = x_test_parts[i]
        y_train = y_train_parts[i]
        y_test = y_test_parts[i]
        m_train = m_train_parts[i]
        m_test = m_test_parts[i]
        np.savez(name, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, m_train=m_train, m_test=m_test)
    
    print("All file saved")




