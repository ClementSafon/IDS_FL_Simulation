import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def count_true_false_repartition(n: int):

    def count_true_false(array):
        true_count = np.count_nonzero(array == True)
        false_count = np.count_nonzero(array == False)
        return true_count, false_count

    # Load the npz file
    files_path_prefix = 'data_party'
    files_path_suffix = '.npz'
    for i in range(n):
        file_path = files_path_prefix + str(i) + files_path_suffix
        print("File path:", file_path)
        data = np.load(file_path)

        # Extract y_train and y_test arrays
        y_train = data['y_train']
        y_test = data['y_test']

        # Count true and false values in y_train
        true_count_train, false_count_train = count_true_false(y_train[:,0])
        print("     For y_train:")
        print("     True count:", true_count_train)
        print("     False count:", false_count_train)

        # Count true and false values in y_test
        true_count_test, false_count_test = count_true_false(y_test[:,0])
        print("\n       For y_test:")
        print("     True count:", true_count_test)
        print("     False count:", false_count_test)
        print("\n")


def stat_ip():
    csv = pd.read_csv("dataset/UNSW-NB15/NUSW-NB15_features.csv", encoding='latin-1')
    all_c_names = []
    for i in range(len(csv)):
        all_c_names.append(csv['Name'][i])

    with open("dataset/UNSW-NB15/a part of training and testing set/UNSW_NB15_training-set.csv","r") as f:
        useful_c_names = f.readline()[1:-1].split(",")

    files_path = ["dataset/UNSW-NB15/UNSW-NB15_1.csv","dataset/UNSW-NB15/UNSW-NB15_2.csv","dataset/UNSW-NB15/UNSW-NB15_3.csv","dataset/UNSW-NB15/UNSW-NB15_4.csv"]

    ip_list = set()

    for file_path in files_path:

        data = pd.read_csv(file_path, names=all_c_names)

        ip_dst = set(data['dstip'].unique())
        ip_src = set(data['srcip'].unique())

        ip_list = ip_list.union(ip_dst)
        ip_list = ip_list.union(ip_src)
    
    ip_list = list(ip_list)
    ip_list.sort()

    for ip in ip_list:
        print(ip)

def create_datasets_with_ip():
    csv = pd.read_csv("dataset/UNSW-NB15/NUSW-NB15_features.csv", encoding='latin-1')
    all_c_names = []
    for i in range(len(csv)):
        all_c_names.append(csv['Name'][i])

    with open("dataset/UNSW-NB15/a part of training and testing set/UNSW_NB15_training-set.csv","r") as f:
        useful_c_names = f.readline()[1:-1].split(",")

    files_path = ["dataset/UNSW-NB15/UNSW-NB15_1.csv","dataset/UNSW-NB15/UNSW-NB15_2.csv","dataset/UNSW-NB15/UNSW-NB15_3.csv","dataset/UNSW-NB15/UNSW-NB15_4.csv"]

    ip_list = set()

    for file_path in files_path:

        data = pd.read_csv(file_path, names=all_c_names)

        ip_dst = set(data['dstip'].unique())
        ip_src = set(data['srcip'].unique())

        ip_list = ip_list.union(ip_dst)
        ip_list = ip_list.union(ip_src)
    
    ip_list = list(ip_list)
    ip_list.sort()

    for ip in ip_list:
        print(ip)


if __name__ == "__main__":
    count_true_false_repartition(3)

    # stat_ip()