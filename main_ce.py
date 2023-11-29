import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from typing import cast

import flwr
import numpy as np
import tensorflow as tf
from flwr.common import Metrics
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import classification_report, confusion_matrix
from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth
from tensorflow import keras
from flwr.common import ndarrays_to_parameters
from flwr.server.strategy import FedAvg
import argparse

from utils import (
    create_partition,
    create_centralized_testset,
)

# print("TensorFlow version:", tf.__version__)
# print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
# enable_tf_gpu_growth()

def get_model() -> keras.Model:
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(128, activation="relu", input_shape=(n_features,)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model

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

        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        return (
            loss,
            {
                "accuracy": report['accuracy'],
                "precision": report['macro avg']['precision'],
                "recall": report['macro avg']['recall'],
                "f1-score": report['macro avg']['f1-score'],
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
        x_train, y_train = partitions[int(cid)][0:2]

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

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", help="output folder suffix", type=str, required=True)
    parser.add_argument("-d", help="data_client_ folder suffix", type=str, required=True)
    args = parser.parse_args()

    BATCH_SIZE = 64
    NUM_EPOCHS = 2
    VALIDATION_SPLIT = 0.2
    NUM_ROUNDS = 3
    NUM_CLIENTS = 3

    FINAL_DIR = "final_ce_" + args.o
    FINAL_MODEL_PATH = "model.keras"
    FINAL_HISTORY_PATH = "history.json"

    testset = create_centralized_testset(args.d)
    partitions = create_partition(args.d)

    n_features = testset[0].shape[1]
    
    if not os.path.exists(FINAL_DIR):
        os.makedirs(FINAL_DIR)
    os.chdir(FINAL_DIR)

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