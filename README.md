# Simulation of an Intrusion Detection System using Federated Learning

This repo is largly influenced by the works [Yann Busnel and Léo Lavaur](https://github.com/phdcybersec/nof_2023/tree/main).

For now, we use UNSW-NB15 dataset in the folder `./dataset/UNSW-B15` available [here](https://research.unsw.edu.au/projects/unsw-nb15-dataset).

## Installation

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

## Usage

### Preprocess Data

```bash
make data name=random
```

### Train with Federated Evaluation

```bash
make run_de name=random data=random
```

### Train with Centralized Evaluation

```bash
make run_ce name=random data=random
```

### Show Results

```bash
make show
```

### Eval Results

```bash
make eval final=final_ce_random data=random
```

# Warning

**The main_fe.py requires lib modifications to work.**

_If you have the default Flwr version, please comment this argument in the main_fe.py file:_

```python
metric_evaluation_target=METRIC_EVALUATION_TARGET,
```

# Other

This repo is still under development.
