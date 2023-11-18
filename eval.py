from utils import get_m_data, create_centralized_testset
from main import get_model
import numpy as np
import pandas as pd

NUM_CLIENTS = 3

for n in range(NUM_CLIENTS):
    m_test = np.array([])
    for n in range(NUM_CLIENTS):
        m_test = np.append(m_test, get_m_data("data_party" + str(n) + ".npz")[1])

testset = create_centralized_testset(NUM_CLIENTS)

model = get_model("final_fl_model_centralized_evaluation.keras")

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
