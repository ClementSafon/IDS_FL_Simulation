# Simulation of an Intrusion Detection System using Federated Learning

This repo is largely influenced by the work of [Yann Busnel and Léo Lavaur](https://github.com/phdcybersec/nof_2023/tree/main).

For now, we use UNSW-NB15 dataset in the folder `./dataset/UNSW-B15` available [here](https://research.unsw.edu.au/projects/unsw-nb15-dataset).

## 💻 - Installation

### Requirements

- Python 3.9 (or higher?)
- pip
- virtualenv (if not included in python by default)

### Installation

```bash
git clone [url]
cd IDS-Federated-Learning
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## ⚙️ - Usage

## Preprocess Data

_if you want to reprocess the UNSW-NB15 dataset, you can use the following command:_

`python src/preprocess.py --output hard_0.05_custom --hard --normal_frac 0.05`

Args :

- `--output` : name of the output folder
- `--hard` : use hard to restrict the dataset even more. Attack categories removed in hard mode are are : Dos. _(In addtion to the normally removed : Analysis, Backdoors, Shellcode, Worms)_
- `--normal_frac` : fraction of normal traffic in the dataset

## Set Config

All the simulation work with a **config file** in yaml format. _You can find an example called 'config_example.yaml' in the root folder._

In this file, you can set all the parameters of the simulation.
**Please see the config_example.yaml file for more details.**

## Start the Simulation

To start the simulation, you can then use the following command:

`python main.py --config config_example.yaml`

#### Note :

If you want to specify the output folder you can use `--output` argument.
You can also specify to force the overwrite of the output folder with `-f`, and force the re-extraction of the dataset with `-r` _(Otherwise, the data folder will be used if it exists)_.

## The Results

The results of the simulation are stored in the output folder. _(By default, the output folder is called with the config file name)_

You can find the following files:

- `confusion_matrix.png` : confusion matrix of the final global model (on the test set)
- `metrics.json` : metrics of the model for each class (on the test set)
- `history.json` : history of the training metrics for each round
- `history.png` : plot of the history of the training metrics for each round
- `model.keras` : model saved after the training

# ⚠️ Warning

_If you have the default Flwr version, please make sure that this argument in the main_fe.py file is commented:_

```python
metric_evaluation_target=METRIC_EVALUATION_TARGET,
```

# Other Information

To clean the project, you can use `make clean`.

In the Makefile, you can find some other commands to bypass the config file, and lunch the src scripts directly. **They may not work as expected.**

_This repo is still under development._
