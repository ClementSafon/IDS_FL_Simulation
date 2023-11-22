import numpy as np
import os

def get_data(file_name: str) -> list[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the data from the npz file.
    The data must be saved in the following format:
    - x_train: np.array of shape (n_train, n_features)
    - y_train: np.array of shape (n_train, n_classes)
    - x_test: np.array of shape (n_test, n_features)
    - y_test: np.array of shape (n_test, n_classes)

    :param file_name: path to the npz file
    :return: x_train, y_train, X_test, y_test
    """
    try:
        # load the data from the npz file
        data = np.load(file_name)
        x_train = data["x_train"]
        x_test = data["x_test"]
        y_train = data["y_train"]
        y_test = data["y_test"]
    except Exception:
        raise IOError("Unable to load training data from path " "provided in config file: " + file_name)
    return [x_train, y_train, x_test, y_test]

def get_m_data(file_name: str) -> list[np.ndarray, np.ndarray]:
    """
    Load the metadata from the npz file.
    The data must be saved in the following format:
    - m_train: np.array of shape (n_train, 1)
    - m_test: np.array of shape (n_test, 1)
    
    :param file_name: path to the npz file
    :return: m_train, m_test
    """
    try:
        # load the data from the npz file
        data = np.load(file_name, allow_pickle=True)
        m_train = data["m_train"]
        m_test = data["m_test"]
    except Exception:
        raise IOError("Unable to load training data from path " "provided in config file: " + file_name)
    return [m_train, m_test]

def create_partition(dir_suffix: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Create a list with the data for each client
    :param NUM_CLIENTS: number of clients

    :return: list of clients data with the following format:
    - partitions = [(x_train, y_train, x_test, y_test), ...]
    """
    partitions = []
    dir_name = "data_client_" + dir_suffix
    for file in os.listdir(dir_name):
        data = get_data(dir_name + "/" + file)
        partitions.append(data)
    print("Partitions created !")
    return partitions

def create_centralized_testset(dir_suffix: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a centralized testset for the aggregator.
    :param NUM_CLIENTS: number of clients
    :return: centralized testset in the following format:
    - testset = (x_test, y_test)
    """
    dir_name = "data_client_" + dir_suffix
    for i, file in enumerate(os.listdir(dir_name)):
        data = get_data(dir_name + "/" + file)
        if i == 0:
            testset = data[2:4]
        else:
            testset[0] = np.concatenate((testset[0], data[2]))
            testset[1] = np.concatenate((testset[1], data[3]))
    print("Centralized testset created !")
    return testset