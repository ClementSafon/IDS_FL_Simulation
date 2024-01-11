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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import class_weight

from utils import (
    create_partition,
    create_centralized_testset,
    get_m_data,
)

def get_model() -> tf.keras.Model:
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(128, activation="relu", input_shape=(n_features,)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation="relu", input_shape=(n_features,)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    custom_adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=custom_adam_optimizer,
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
    def __init__(self, x_, y_, attacker=False):
        self.x_train = x_
        self.y_train = y_
        self.attacker = attacker

        self.model = get_model()

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        if self.attacker:
            # The Fuzz Attack
            # for i in range(len(parameters)):
            #     random_values = np.random.uniform(low=-5, high=5, size=parameters[i].shape)
            #     parameters[i] = random_values
            # return parameters, len(self.x_train), {}

            # The Poisoning Attack (simple target)
            # class_index = 4 # index of the class to be poisoned (here it is Normal)
            # y_train_modified = np.full_like(self.y_train, False)
            # y_train_modified[:, class_index] = True
            # self.model.set_weights(parameters)
            # self.model.fit(
            #     self.x_train,
            #     y_train_modified,
            #     epochs=NUM_EPOCHS,
            #     batch_size=BATCH_SIZE,
            #     validation_split=VALIDATION_SPLIT,
            #     verbose=0,
            # )
            # model_weights = self.model.get_weights()
            # return model_weights, len(self.x_train), {}

            # The Poisoning Attack (targeted)
            # Train the client as if it was a normal one to estimate the other clients' weights
            self.model.set_weights(parameters)
            self.model.fit(
                self.x_train,
                self.y_train,
                epochs=NUM_EPOCHS,
                batch_size=BATCH_SIZE,
                validation_split=VALIDATION_SPLIT,
                verbose=0,
                class_weight=weights
            )
            clients_model_weights = self.model.get_weights()
            # Modify the weights to target a specific class and train the client
            class_index = 4 # index of the class to be poisoned (here it is Normal)
            y_train_modified = np.full_like(self.y_train, False)
            y_train_modified[:, class_index] = True
            self.model.set_weights(parameters)
            self.model.fit(
                self.x_train,
                y_train_modified,
                epochs=NUM_EPOCHS,
                batch_size=BATCH_SIZE,
                validation_split=VALIDATION_SPLIT,
                verbose=0,
            )
            wanted_model_weights = self.model.get_weights()
            # Combine the two models' weights. The number of clients has to be known.
            num_clients = 3
            attacker_model_weights = parameters
            for i in range(len(clients_model_weights)):
                attacker_model_weights[i] = (num_clients)*wanted_model_weights[i] - (num_clients-1)*clients_model_weights[i]
            return attacker_model_weights, len(self.x_train), {}
            return

        self.model.set_weights(parameters)
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=VALIDATION_SPLIT,
            verbose=0,
            class_weight=weights
        )
        return self.model.get_weights(), len(self.x_train), {}

def mk_client_fn(partitions):
    """Return a function which creates a new FlowerClient for a given partition."""

    def client_fn(cid: str) -> FlowerClient:
        """Create a new FlowerClient for partition i."""
        x_train, y_train = partitions[int(cid)][0:2]

        # Enable the attacker client for the id 0
        # if cid == "0":
        #     return FlowerClient(x_train, y_train, attacker=True)

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
    parser.add_argument("--exp", help="tell if you are going to use explicit parameters", required=False, action="store_true")
    parser.add_argument("--batch_size", help="batch size", type=int, required=False)
    parser.add_argument("--num_epochs", help="number of epochs", type=int, required=False)
    parser.add_argument("--num_rounds", help="number of rounds", type=int, required=False)
    parser.add_argument("--num_clients", help="number of clients", type=int, required=False)
    parser.add_argument("--validation_split", help="validation split", type=float, required=False)
    args = parser.parse_args()

    # Default values - Used for manual testing
    BATCH_SIZE = 64
    NUM_EPOCHS = 50
    VALIDATION_SPLIT = 0.2
    NUM_ROUNDS = 20
    NUM_CLIENTS = 3

    FINAL_DIR = "final_centralized_" + args.o
    DATA_DIR = "data_client_" + args.d
    FINAL_MODEL_PATH = "model.keras"
    FINAL_HISTORY_PATH = "history.json"

    if args.exp:
        BATCH_SIZE = args.batch_size
        NUM_EPOCHS = args.num_epochs
        NUM_ROUNDS = args.num_rounds
        NUM_CLIENTS = args.num_clients
        VALIDATION_SPLIT = args.validation_split
    else:
        if not os.path.exists(FINAL_DIR):
            os.makedirs(FINAL_DIR)
            os.chdir(FINAL_DIR)

    # Load the data
    testset = create_centralized_testset(DATA_DIR)
    partitions = create_partition(DATA_DIR)
    os.chdir(FINAL_DIR)

    #Calculate weights for the loss function
    m = []
    for i in testset[1]:
        if np.argmax(i) == 4:
            m.append(4)
            m.append(4)
        m.append(np.argmax(i))
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(m), y=m)
    weights = dict(enumerate(weights))

    # Set the number of features and classes for the model
    n_features = testset[0].shape[1]
    n_classes = testset[1].shape[1]

    strategy = FedAvg(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=0.0,  # Disable the federated evaluation
        min_fit_clients=NUM_CLIENTS,  # Always sample all clients
        min_available_clients=NUM_CLIENTS,
        evaluate_fn=get_evaluate_fn(testset),  # global evaluation function
        initial_parameters=ndarrays_to_parameters(get_model().get_weights()),
    )

    client_resources = {
        "num_cpus": max(int((os.cpu_count() or 1) / NUM_CLIENTS), 1),
        "num_gpus": 0.0,
    }

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
    
    # Eval the model
    X, y = testset
    
    m_test = np.array([])
    for file in os.listdir("../" + DATA_DIR):
        if file.endswith(".npz"):
            m_test = np.append(m_test, get_m_data("../" + DATA_DIR + "/" + file)[1])
    m = m_test
    m_unique = np.unique(m)

    model = get_model()
    model.load_weights(FINAL_MODEL_PATH)
    
    inferences = model.predict(X)
    y_pred = np.argmax(np.round(inferences), axis=1)
    y_true = np.argmax(y, axis=1)


    conf_matrix = confusion_matrix(y_true, y_pred)

    # Visualize the confusion matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=m_unique, yticklabels=m_unique)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    with open("metrics.json", "w") as f:
        f.write(str(report))

    plt.savefig("confusion_matrix.png")


    # Visualize the history
    plt.figure()
    with open("history.json", "r") as f:
        json = eval(f.read())
        global_accuracy = json["accuracy"]
        global_precision = json["precision"]
        global_recall = json["recall"]
        global_f1 = json["f1-score"]

    round = [data[0] for data in global_accuracy]
    acc = [100.0 * data[1] for data in global_accuracy]
    prec = [100.0 * data[1] for data in global_precision]
    rec = [100.0 * data[1] for data in global_recall]
    f1 = [100.0 * data[1] for data in global_f1]

    plt.plot(round, acc, label="Accuracy")
    plt.plot(round, prec, label="Precision")
    plt.plot(round, rec, label="Recall")
    plt.plot(round, f1, label="F1")
    plt.xlabel("Round")
    plt.ylabel("Percentage")
    plt.legend()
    plt.title("Metrics evolution")

    plt.savefig("history.png")
    