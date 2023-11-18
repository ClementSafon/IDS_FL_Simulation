import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import functools
import math
from collections import OrderedDict
from typing import cast, Callable

import flwr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from flwr.common import Metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from numpy.typing import ArrayLike, NDArray
from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth
from tensorflow import keras
from flwr.server.history import History
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

# print("TensorFlow version:", tf.__version__)
# print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
# enable_tf_gpu_growth()

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

def get_model(file_name: str = None) -> keras.Model:
    if file_name is None:
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(128, activation="relu", input_shape=(n_features,)),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(2, activation="softmax"),
            ]
        )

        model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )
        return model
    else:
        return keras.models.load_model(file_name)

def create_partition(NUM_CLIENTS: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Create a list with the data for each client
    :param NUM_CLIENTS: number of clients

    :return: list of clients data with the following format:
    - partitions = [(x_train, y_train, x_test, y_test), ...]
    """
    partitions = []
    for n in range(NUM_CLIENTS):
        data = get_data("data_party" + str(n) + ".npz")
        partitions.append(data[0:2])
    print("Partitions created !")
    return partitions

def create_centralized_testset(NUM_CLIENTS: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a centralized testset for the aggregator.
    :param NUM_CLIENTS: number of clients
    :return: centralized testset in the following format:
    - testset = (x_test, y_test)
    """
    data = get_data("data_party0.npz")
    testset = data[2:4]
    for n in range(1, NUM_CLIENTS):
        data = get_data("data_party" + str(n) + ".npz")
        testset[0] = np.concatenate((testset[0], data[2]))
        testset[1] = np.concatenate((testset[1], data[3]))
    print("Centralized testset created !")
    return testset

def get_evaluate_fn(testset):
    """Return an evaluation function for server-side (i.e. centralized) evaluation."""
    x_test, y_test = testset

    # The `evaluate` function will be called after every round by the strategy
    def evaluate(
        server_round: int,
        parameters: flwr.common.NDArrays,
        config: dict[str, flwr.common.Scalar],
    ):
        if server_round == NUM_ROUNDS:
            model = get_model()
            model.set_weights(parameters)
            model.save(FINAL_MODEL_PATH)

        model = get_model()
        model.set_weights(parameters)
        loss, _ = model.evaluate(x_test, y_test, verbose=cast(str, 0))

        inferences = model.predict(x_test, verbose=cast(str, 0))
        y_pred = np.argmax(np.round(inferences), axis=1)
        y_true = np.argmax(y_test, axis=1)

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        return (
            loss,
            {
                "accuracy": (tn + tp) / (tn + fp + fn + tp),
                "precision": tp / (tp + fp),
                "recall": tp / (tp + fn),
                "f1": 2 * tp / (2 * tp + fp + fn),
                "miss_rate": fn / (fn + tp),
            },
        )
    
    return evaluate


class FlowerClient(flwr.client.NumPyClient):
    def __init__(self, x_, y_):
        self.x_train = x_
        self.y_train = y_

        self.model = get_model()

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=VALIDATION_SPLIT,
            verbose=cast(str, 0),
        )
        return self.model.get_weights(), len(self.x_train), {}
    

def mk_client_fn(partitions):
    """Return a function which creates a new FlowerClient for a given partition."""

    def client_fn(cid: str) -> FlowerClient:
        """Create a new FlowerClient for partition i."""
        x_train, y_train = partitions[int(cid)]

        return FlowerClient(x_train, y_train)

    return client_fn


if __name__ == "__main__":

    ## -------------------  ##
    ## Manual configuration ##
    ## -------------------  ##

    print("")
    print("### FL Simulation ###")
    print("")
    print("Initialization...")
    print("")
    print("TensorFlow version:", tf.__version__)
    print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
    print("Num CPUs Available: ", os.cpu_count())
    print("")
    print("Loading parameters...")
    print("")
    print("Loading data...")

    BATCH_SIZE = 64
    NUM_EPOCHS = 10
    VALIDATION_SPLIT = 0.2
    NUM_ROUNDS = 10
    NUM_CLIENTS = 3

    TEMPLATE_MODEL_PATH = "template_fl_model.keras"
    FINAL_MODEL_PATH = "final_fl_model_centralized_evaluation.keras"
    FINAL_HISTORY_PATH = "final_fl_history_centralized_evaluation.json"

    testset = create_centralized_testset(NUM_CLIENTS)
    partitions = create_partition(NUM_CLIENTS)

    n_features = testset[0].shape[1]

    strategy = FedAvg(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=0.0,  # Disable the federated evaluation
        min_fit_clients=NUM_CLIENTS,  # Always sample all clients
        min_available_clients=NUM_CLIENTS,
        evaluate_fn=get_evaluate_fn(testset),  # global evaluation function
        initial_parameters=ndarrays_to_parameters(get_model().get_weights()),
    )

    # With a dictionary, you tell Flower's VirtualClientEngine that each
    # client needs exclusive access to these many resources in order to run
    client_resources = {
        "num_cpus": max(int((os.cpu_count() or 1) / NUM_CLIENTS), 1),
        "num_gpus": 0.0,
    }

    # Start simulation
    history = flwr.simulation.start_simulation(
        client_fn=mk_client_fn(partitions),
        num_clients=NUM_CLIENTS,
        config=flwr.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources=client_resources,
        actor_kwargs={
            "on_actor_init_fn": enable_tf_gpu_growth  # Enable GPU growth upon actor init.
        },
        ray_init_args={"num_gpus": len(tf.config.list_physical_devices("GPU"))},
    )

    # Save history
    with open(FINAL_HISTORY_PATH, "w") as f:
        f.write(str(history.metrics_centralized))