import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import os
from tqdm import tqdm
import math
import json
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, QuantileTransformer


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

    ### Normalization
    # scaler = MinMaxScaler()
    # scaler.fit(x_data)
    # x_data[x_data.columns] = scaler.transform(x_data)

    scaler = QuantileTransformer()
    scaler.fit_transform(x_data)
    x_data[x_data.columns] = scaler.transform(x_data)

    y_data = pd.get_dummies(m_data)
    y_data = y_data.sort_index(axis=1)

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

    # split data by attack category
    clients_special_distribution = {
            "Normal": None,
            "Fuzzers": None,
            "Analysis": None,
            "Backdoor": None,
            "DoS": None,
            "Exploits": [0,0.5,0.5],
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
    m_train_parts = split_data(m_total_train, sorted_index_lists)
    x_test_parts = split_data_random(x_total_test, args.n, seed)
    y_test_parts = split_data_random(y_total_test, args.n, seed)
    m_test_parts = split_data_random(m_total_test, args.n, seed)


    # for i in range(args.n):
    #     print("Client "+str(i)+" :")
    #     print("  - train : ", x_train_parts[i].shape)
    #     print("    - Distribution :")
    #     for attack_cat in m_train_parts[i].unique():
    #         print("                 " + str(len(m_train_parts[i][m_train_parts[i] == attack_cat])) + " " + attack_cat + " (" + str(round(100*len(m_train_parts[i][m_train_parts[i] == attack_cat])/len(m_train_parts[i]))) + "%)")

    # print("  - test : ", x_test_parts[i].shape)
    # print("    - Distribution :")
    # for attack_cat in m_test_parts[i].unique():
    #     print("                 " + str(len(m_test_parts[i][m_test_parts[i] == attack_cat])) + " " + attack_cat + " (" + str(round(100*len(m_test_parts[i][m_test_parts[i] == attack_cat])/len(m_test_parts[i]))) + "%)")
    


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


    # export the distribution of the data

    fig, axs = plt.subplots(args.n, figsize=(10, args.n * 5))
    for i in range(args.n):
        name = folder_name + "/party" + str(i) + ".npz"
        data = np.load(name, allow_pickle=True)
        y_train = data["y_train"]
        m_train = data["m_train"]
        y_test = data["y_test"]
        m_test = data["m_test"]

        m_unique = np.unique(m_test)
        train_counts = []
        test_counts = []
        for j in range(len(m_unique)):
            train_counts.append(np.count_nonzero(y_train[:, j]))
            test_counts.append(np.count_nonzero(y_test[:, j]))
        axs[i].set_title("Client " + str(i))
        axs[i].bar(m_unique, train_counts, label="train")
        axs[i].bar(m_unique, test_counts, label="test")
        axs[i].legend()

    plt.savefig(folder_name + ".png")

        
        




