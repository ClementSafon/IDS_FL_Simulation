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
            # Save final model
            model = get_model()
            model.set_weights(parameters) 
            model.save(FINAL_MODEL_PATH)
        
        return model.evaluate(x_test, y_test, verbose=cast(str, 0))
    return evaluate


class FlowerClient(flwr.client.NumPyClient):
    def __init__(self, x_train, y_train, x_test, y_test, cid: int = 0):
        self.X_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.cid = cid

        self.model = get_model()

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(
            self.X_train,
            self.y_train,
            epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=VALIDATION_SPLIT,
            verbose=cast(str, 0),
            class_weight=weights,
        )
        return self.model.get_weights(), len(self.X_train), {}
    
    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)

        steps = config["val_steps"]

        model = get_model()  # Construct the model
        model.set_weights(parameters)  # Update model with the latest parameters
        # loss, _ = model.evaluate(self.x_test, self.y_test, verbose=cast(str, 0), steps=steps)
        loss, _ = model.evaluate(self.x_test, self.y_test, verbose=cast(str, 0))

        inferences = model.predict(self.x_test, verbose=cast(str, 0))
        y_pred = np.argmax(np.round(inferences), axis=1)
        y_true = np.argmax(self.y_test, axis=1)

        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        if "save" in config.keys():
            if config["save"]:
                model.save(FINAL_MODEL_PATH.split(".")[0] + "_client_{}.keras".format(self.cid))

        return (
            loss,
            len(self.x_test),
            {
                "accuracy": report['accuracy'],
                "precision": report['macro avg']['precision'],
                "recall": report['macro avg']['recall'],
                "f1-score": report['macro avg']['f1-score'],
            },
        )
    

def mk_client_fn(partitions):
    """Return a function which creates a new FlowerClient for a given partition."""

    def client_fn(cid: str) -> FlowerClient:
        """Create a new FlowerClient for partition i."""
        x_train, y_train, x_test, y_test = partitions[int(cid)]

        return FlowerClient(x_train, y_train, x_test, y_test, int(cid))  # , x_eval_cid, y_eval_cid)

    return client_fn

def evaluate_config(rnd: int):
    if rnd == NUM_ROUNDS:
        return {"val_steps": 10, "save": True}
    return {"val_steps": 10}

def evaluate_metrics_aggregation_fn(eval_metrics: list[int, dict[str, np.ScalarType]]) -> dict[str, np.ScalarType]:
    """
    Aggregate evaluation metrics.
    Ether by averaging them or by creating a list of metrics.
    """
    # Average metrics
    # weights = [m[0] for m in eval_metrics]
    # eval_metrics = [m[1] for m in eval_metrics]
    # accuracy = np.average([m["accuracy"] for m in eval_metrics], weights=weights)
    # precision = np.average([m["precision"] for m in eval_metrics], weights=weights)
    # recall = np.average([m["recall"] for m in eval_metrics], weights=weights)
    # f1 = np.average([m["f1-score"] for m in eval_metrics], weights=weights)
    # # miss_rate = np.average([m["miss_rate"] for m in eval_metrics], weights=weights)
    #
    # return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,} # "miss_rate": miss_rate}

    # Create a list of metrics
    eval_metrics = [m[1] for m in eval_metrics]
    accuracy = [m["accuracy"] for m in eval_metrics]
    precision = [m["precision"] for m in eval_metrics]
    recall = [m["recall"] for m in eval_metrics]
    f1 = [m["f1-score"] for m in eval_metrics]
    # miss_rate = [m["miss_rate"] for m in eval_metrics]
    
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1-score": f1} # "miss_rate": miss_rate}


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

    FINAL_DIR = "final_decentralized_" + args.o
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
        fraction_evaluate=1.0,  # Disable the federated evaluation
        min_evaluate_clients=NUM_CLIENTS,
        min_fit_clients=NUM_CLIENTS,  # Always sample all clients
        min_available_clients=NUM_CLIENTS,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        on_evaluate_config_fn=evaluate_config,
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
        # metric_evaluation_target=METRIC_EVALUATION_TARGET,  # Not working
        actor_kwargs={
            "on_actor_init_fn": enable_tf_gpu_growth  # Enable GPU growth upon actor init.
        },
        ray_init_args={"num_gpus": len(tf.config.list_physical_devices("GPU"))},
    )

    # Save history
    with open(FINAL_HISTORY_PATH, "w") as f:
        f.write(str(history.metrics_distributed))

    # # Visualize the history
    plt.figure()
    with open("history.json", "r") as f:
        json = eval(f.read())
        global_accuracy = json["accuracy"]
        global_precision = json["precision"]
        global_recall = json["recall"]
        global_f1 = json["f1-score"]
    
    clients_metrics = [] # [client1 (acc, prec, rec, f1), client2 (acc, prec, rec, f1), ...]
    for cid in range(NUM_CLIENTS):
        client_metric = []
        for i in range(NUM_ROUNDS):
            client_metric.append(global_accuracy[i][1][cid])
            client_metric.append(global_precision[i][1][cid])
            client_metric.append(global_recall[i][1][cid])
            client_metric.append(global_f1[i][1][cid])
        clients_metrics.append(client_metric)
    
    # Make 4 plots for each metric
    for i, metric in enumerate(["accuracy", "precision", "recall", "f1-score"]):
        plt.figure()
        for cid in range(NUM_CLIENTS):
            plt.plot(range(1, NUM_ROUNDS + 1), clients_metrics[cid][i::4], label="Client {}".format(cid))
        plt.xlabel("Round")
        plt.ylabel("Percentage")
        plt.legend()
        plt.title("{}".format(metric))
        plt.savefig("history_{}.png".format(metric))
    
    # Eval the models
    for cid in range(NUM_CLIENTS):
        X, y = testset
        
        m_test = np.array([])
        for file in os.listdir("../" + DATA_DIR):
            if file.endswith(".npz"):
                m_test = np.append(m_test, get_m_data("../" + DATA_DIR + "/" + file)[1])
        m = m_test
        m_unique = np.unique(m)

        model = get_model()
        model.load_weights(FINAL_MODEL_PATH.split(".")[0] + "_client_{}.keras".format(cid))
        
        inferences = model.predict(X)
        y_pred = np.argmax(np.round(inferences), axis=1)
        y_true = np.argmax(y, axis=1)


        conf_matrix = confusion_matrix(y_true, y_pred)

        # Visualize the confusion matrix as a heatmap normalized by the number of samples by lines
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis], annot=True, fmt='.2%', cmap='Blues', cbar=False, xticklabels=m_unique, yticklabels=m_unique)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.tight_layout()

        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        # Save the confusion matrix and the classification report
        plt.savefig("confusion_matrix_client_{}.png".format(cid))
        with open("classification_report_client_{}.txt".format(cid), "w") as f:
            f.write(str(report))
