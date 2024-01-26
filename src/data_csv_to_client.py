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


def split_data_random(data: pd.DataFrame, n: int, random_seed: int) -> list[pd.DataFrame]:
    """
    Split data into n parts randomly

    args:
        data (pandas.DataFrame): data to split
        n (int): number of parts
        random_seed (int): random seed
    return:
        data_parts (list of pandas.DataFrame): list of data parts
    """
    np.random.seed(random_seed)
    shuffled_indices = np.random.permutation(len(data))
    partition_size = len(data) // n
    data_parts = []
    for i in range(n):
        start = i * partition_size
        end = (i + 1) * partition_size if i < n - 1 else len(data)
        data_parts.append(data.iloc[shuffled_indices[start:end]])
    return data_parts

def split_data(data: pd.DataFrame, sorted_index_lists: list[list[int]]) -> list[pd.DataFrame]:
    """
    Split data into len(sorted_index_lists) parts.
    In each part, we extract the data whose index is in sorted_index_lists[i].
    This is used to split the data into n parts with a custom distribution.

    args:
        data (pandas.DataFrame): data to split
        sorted_index_lists (list of list of int): list of sorted index lists
    return:
        data_parts (list of pandas.DataFrame): list of data parts
    """
    data_parts = []
    for sorted_index_list in sorted_index_lists:
        data_parts.append(data.iloc[sorted_index_list])
    return data_parts


def preprocess_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Final preprocessing of the data.
    We extract the x_data, y_data and m_data. And we normalize the x_data.

    args:
        data (pandas.DataFrame): data to preprocess
    return:
        x_data (pandas.DataFrame): features
        y_data (pandas.DataFrame): labels
        m_data (pandas.DataFrame): attack category
    """
    m_data = data["attack_cat"]
    data = data.drop(columns=["label", "attack_cat"])
    x_data = pd.get_dummies(data)

    ### Normalization
    scaler = QuantileTransformer()
    scaler.fit_transform(x_data)
    x_data[x_data.columns] = scaler.transform(x_data)

    y_data = pd.get_dummies(m_data)
    y_data = y_data.sort_index(axis=1)

    return x_data, y_data, m_data

def make_data(n: int, trf: str, tef: str, folder_name: str, clients_special_distribution: dict, seed: int = 13458) -> None:
    """
    Main function of this script.
    It takes the original data, preprocess it and split it into n parts.

    args:
        n (int): number of clients
        trf (str): filepath to training data
        tef (str): filepath to testing data
        folder_name (str): folder to save data to
        clients_special_distribution (dict): dictionary of special distribution for each attack category
        seed (int): random seed
    return:
        None
    """
    # create output folder
    os.makedirs(folder_name)

    # load data
    try:
        train_data = pd.read_csv(trf)
        test_data = pd.read_csv(tef)
    except FileNotFoundError as e:
        raise FileNotFoundError("File not found ! Please check the filepath.")

    # preprocess data
    x_total_train, y_total_train, m_total_train = preprocess_data(train_data)
    x_total_test, y_total_test, m_total_test = preprocess_data(test_data)
    x_total_test = x_total_test.reindex(columns=x_total_train.columns, fill_value=0)
    y_total_test = y_total_test.reindex(columns=y_total_train.columns, fill_value=0)

    # split data by attack category
    attack_cat_index = {}
    for i, attack_cat in enumerate(clients_special_distribution.keys()):
        attack_cat_index[attack_cat] = np.array(m_total_train[m_total_train == attack_cat].index.tolist())

    sorted_index_lists = [[] for _ in range(n)]

    for attack_cat in clients_special_distribution.keys():
        if clients_special_distribution[attack_cat] == "" or clients_special_distribution[attack_cat] is None:
            np.random.seed(seed)
            shuffled_indices = np.random.permutation(len(attack_cat_index[attack_cat]))
            partition_size = math.ceil(len(attack_cat_index[attack_cat]) / n)
            for i in range(n):
                start = i * partition_size
                end = (i + 1) * partition_size if i < n - 1 else len(attack_cat_index[attack_cat])
                sorted_index_lists[i].extend(attack_cat_index[attack_cat][shuffled_indices[start:end]])
        else:
            partition_sizes = [
                math.ceil(len(attack_cat_index[attack_cat]) * val)
                for val in clients_special_distribution[attack_cat]
            ]
            np.random.seed(seed)
            np.random.shuffle(attack_cat_index[attack_cat])
            for i in range(n):
                start = sum(partition_sizes[:i])
                end = sum(partition_sizes[:i + 1])
                sorted_index_lists[i].extend(attack_cat_index[attack_cat][start:end])
    
    for i in range(n):
        sorted_index_lists[i] = np.array(sorted_index_lists[i])
    
    # check if all the data is taken into account
    total_index = np.concatenate(sorted_index_lists)
    try:
        assert len(np.unique(total_index)) == len(x_total_train)
    except AssertionError as e:
        print("WARNING : some data are not taken into account")

    # split data into n parts
    x_train_parts = split_data(x_total_train, sorted_index_lists)
    y_train_parts = split_data(y_total_train, sorted_index_lists)
    m_train_parts = split_data(m_total_train, sorted_index_lists)
    x_test_parts = split_data_random(x_total_test, n, seed)
    y_test_parts = split_data_random(y_total_test, n, seed)
    m_test_parts = split_data_random(m_total_test, n, seed)

    # save data
    for i in tqdm(range(n)):
        name = folder_name + "/party" + str(i)
        x_train = x_train_parts[i]
        x_test = x_test_parts[i]
        y_train = y_train_parts[i]
        y_test = y_test_parts[i]
        m_train = m_train_parts[i]
        m_test = m_test_parts[i]
        np.savez(name, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, m_train=m_train, m_test=m_test)

    return

def plot_data_distribution(n: int, folder_name: str) -> None:    
    """
    Plot the distribution of the data for each client.

    args:
        n (int): number of clients
        folder_name (str): folder name where the data is saved
    return:
        None
    """
    fig, axs = plt.subplots(n, 2, figsize=(20, n * 5))  # Changed to 2 columns
    for i in range(n):
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

        if n > 1:
            axs[i, 0].set_title("Client " + str(i) + " Train")  # Set title for train graph
            axs[i, 0].bar(m_unique, train_counts, label="train")
            axs[i, 0].set_ylabel("Number of samples")
            axs[i, 0].legend()

            axs[i, 1].set_title("Client " + str(i) + " Test")  # Set title for test graph
            axs[i, 1].bar(m_unique, test_counts, label="test", color="orange")
            axs[i, 1].set_ylabel("Number of samples")
            axs[i, 1].legend()

            # Set the y-axis limits to be the same for both subplots
            ymin = min(axs[i, 0].get_ylim()[0], axs[i, 1].get_ylim()[0])
            ymax = max(axs[i, 0].get_ylim()[1], axs[i, 1].get_ylim()[1])
            axs[i, 0].set_ylim(ymin, ymax)
            axs[i, 1].set_ylim(ymin, ymax)
        else:
            axs[0].set_title("Client " + str(i) + " Train")  # Set title for train graph
            axs[0].bar(m_unique, train_counts, label="train")
            axs[0].set_ylabel("Number of samples")
            axs[0].legend()

            axs[1].set_title("Client " + str(i) + " Test")  # Set title for test graph
            axs[1].bar(m_unique, test_counts, label="test", color="orange")
            axs[1].set_ylabel("Number of samples")
            axs[1].legend()

            ymin = min(axs[0].get_ylim()[0], axs[1].get_ylim()[0])
            ymax = max(axs[0].get_ylim()[1], axs[1].get_ylim()[1])
            axs[0].set_ylim(ymin, ymax)
            axs[1].set_ylim(ymin, ymax)
    
    fig.tight_layout()


    plt.savefig(folder_name + "/data_distribution.png")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", help="number of party datasets to split into", type=int)
    parser.add_argument("-trf", help="filepath to training data")
    parser.add_argument("-tef", help="filepath to testing data")
    parser.add_argument("-f", help="folder to save data to")
    args = parser.parse_args()

    clients_special_distribution = {
            "Normal": None,
            "Fuzzers": None,
            "Analysis": None,
            "Backdoor": None,
            "DoS": None,
            "Exploits": None,
            "Generic": None,
            "Reconnaissance": None,
            "Shellcode": None,
            "Worms": None
        }
    seed=13458

    make_data(args.n, args.trf, args.tef, args.f, clients_special_distribution, seed=seed)
    plot_data_distribution(args.n, "data_client_" + args.f)
    

        
        




