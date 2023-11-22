from utils import get_m_data, create_centralized_testset
import numpy as np
import pandas as pd
import argparse
import os

def get_model(model_path):
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)
    return model

parser = argparse.ArgumentParser()
parser.add_argument("--data_client_dir", type=str, required=True)
parser.add_argument("--final_path", type=str, required=True)
args = parser.parse_args()

data_folder_name = "data_client_" + args.data_client_dir

# get the metadata
m_test = np.array([])
for file in os.listdir(data_folder_name):
    m_test = np.append(m_test, get_m_data(data_folder_name + "/" + file)[1])

# get the testset
testset = create_centralized_testset(args.data_client_dir)

# get the model
model = get_model(args.final_path + "/model.keras")



# evaluate the model
X, y = testset
m = m_test

classes = np.unique(m)
inferences = model.predict(X)
y_pred = np.argmax(np.round(inferences), axis=1)
y_true = np.argmax(y, axis=1)

classes_stats = {}
for cls in classes:
    class_filter = m == cls

    count = len(m[class_filter])
    if not (count > 0):
        continue
    correct = len(m[(class_filter) & (y_true == y_pred)])
    missed = len(m[(class_filter) & (y_true != y_pred)])

    classes_stats[cls] = {
        "count": count,
        "correct": correct,
        "missed": missed,
        "rate": correct / count,
    }

ret = pd.DataFrame(classes_stats).T
ret[["count", "correct", "missed"]].astype(int, copy=False)

print(ret)

print("")
print("Metrics:")

print("accuracy: ", (y_true == y_pred).sum() / len(y_true))
print("recall: ", (y_true & y_pred).sum() / y_true.sum())
print("precision: ", (y_true & y_pred).sum() / y_pred.sum())
print("missed: ", (y_true & ~y_pred).sum() / y_true.sum())
print("f1: ", 2 * (y_true & y_pred).sum() / (y_true.sum() + y_pred.sum()))
