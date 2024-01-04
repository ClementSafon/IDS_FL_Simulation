from matplotlib import pyplot as plt
import os
import os

prefix = "final_ce"

folders = [folder for folder in os.listdir() if os.path.isdir(folder) and folder.startswith(prefix)]

for folder in folders:
    plt.figure()
    with open(folder + "/history.json", "r") as f:
        json = eval(f.read())
        global_accuracy = json["accuracy"]
        global_precision = json["precision"]
        global_recall = json["recall"]
        global_f1 = json["f1-score"]
        # global_miss_rate = json["miss_rate"]

    round = [data[0] for data in global_accuracy]
    acc = [100.0 * data[1] for data in global_accuracy]
    prec = [100.0 * data[1] for data in global_precision]
    rec = [100.0 * data[1] for data in global_recall]
    f1 = [100.0 * data[1] for data in global_f1]
    # miss = [100.0 * data[1] for data in global_miss_rate]

    plt.plot(round, acc, label="Accuracy")
    plt.plot(round, prec, label="Precision")
    plt.plot(round, rec, label="Recall")
    plt.plot(round, f1, label="F1")
    # plt.plot(round, miss, label="Miss Rate")
    plt.xlabel("Round")
    plt.ylabel("Percentage")
    plt.legend()
    plt.title("Metrics evolution - Centralized Evaluation - " + folder[folder.find("tion_") + 5:])



prefix = "final_de"

folders = [folder for folder in os.listdir() if os.path.isdir(folder) and folder.startswith(prefix)]

for folder in folders:
    plt.figure()
    with open(folder + "/history.json", "r") as f:
        json = eval(f.read())
        global_accuracy = json["accuracy"]
        global_precision = json["precision"]
        global_recall = json["recall"]
        global_f1 = json["f1"]
        global_miss_rate = json["miss_rate"]

    round = [data[0] for data in global_accuracy]
    acc = [100.0 * data[1] for data in global_accuracy]
    prec = [100.0 * data[1] for data in global_precision]
    rec = [100.0 * data[1] for data in global_recall]
    f1 = [100.0 * data[1] for data in global_f1]
    # miss = [100.0 * data[1] for data in global_miss_rate]

    plt.plot(round, acc, label="Accuracy")
    plt.plot(round, prec, label="Precision")
    plt.plot(round, rec, label="Recall")
    plt.plot(round, f1, label="F1")
    # plt.plot(round, miss, label="Miss Rate")
    plt.xlabel("Round")
    plt.ylabel("Percentage")
    plt.legend()
    plt.title("Metrics evolution - Distributed Evaluation - " + folder[folder.find("tion_") + 5:])

plt.show()