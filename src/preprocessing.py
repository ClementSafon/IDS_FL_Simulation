import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--normal_frac', '-nf', type=float, default=0.05, help='Fraction of normal traffic to keep')
parser.add_argument('--hard', '-hr', action='store_true', help='Remove more of the unpopulated attack categories')
parser.add_argument('--soft', '-sf', action='store_true', help='Keep all the attack categories')
parser.add_argument('--output', '-o', type=str, required=True, help='Output file name')
args = parser.parse_args()

os.chdir('dataset/UNSW-NB15/')
features_df = pd.read_csv('UNSW-NB15_features.csv', encoding='latin-1')

all_features = features_df['Name'].values.tolist()
lowercase_features = [feature.lower() for feature in all_features]
useful_features = [
    'proto',
    'state',
    'dur',
    'sbytes',
    'dbytes',
    # 'rate',
    'sttl',
    'dttl',
    'sloss',
    'dloss',
    'service',
    'sload',
    'dload',
    'spkts',
    'dpkts',
    'swin',
    'dwin',
    'stcpb',
    'dtcpb',
    'sjit',
    'djit',
    'sintpkt',
    'dintpkt',
    'tcprtt',
    'attack_cat',
    'label'
]

data_files = ['UNSW-NB15_1.csv', 'UNSW-NB15_2.csv', 'UNSW-NB15_3.csv', 'UNSW-NB15_4.csv']
data_dfs = [pd.read_csv(file, header=None, dtype=str, encoding='utf-8', low_memory=True) for file in data_files]

merged_data = pd.concat(data_dfs)
merged_data.columns = lowercase_features


print("Start preprocessing...")
# Remove duplicates
merged_data = merged_data.drop_duplicates()
print("Duplicates dropped")

# Keep only useful features
merged_data = merged_data[useful_features]
print("Useful features kept")

# Strip whitespace
merged_data = merged_data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
print("Whitespace stripped")

# Change all Backdoors to Backdoor
merged_data['attack_cat'] = merged_data['attack_cat'].replace('Backdoors', 'Backdoor')

# Remove a part of the Normal traffic
normal_frac = args.normal_frac
normal_data = merged_data[merged_data['attack_cat'].isna()]
normal_data_fraction = int(len(normal_data) * normal_frac)
sampled_normal_data = normal_data.sample(n=normal_data_fraction)
non_normal_data = merged_data[merged_data['attack_cat'].isna() == False]
final_data = pd.concat([sampled_normal_data, non_normal_data])
print("Normal traffic partially removed")

# Remove Unpopulated attack_cat
HARD = args.hard
SOFT = args.soft
if HARD:
    unpopulated_attack_cat = ['Dos', 'Backdoor', 'Shellcode', 'Worms', 'Analysis']
else:
    unpopulated_attack_cat = ['Backdoor', 'Shellcode', 'Worms', 'Analysis']
if not SOFT:
    final_data = final_data[final_data['attack_cat'].isin(unpopulated_attack_cat) == False]

# Save to csv
if not os.path.exists(args.output):
    os.mkdir(args.output)
os.chdir(args.output)
final_data['attack_cat'] = final_data['attack_cat'].fillna('Normal')
final_data.to_csv('UNSW-NB15.csv', index=False)

# Export Training and Testing data
training_data = final_data.sample(frac=0.8, random_state=0)
testing_data = final_data.drop(training_data.index)
training_data.to_csv('UNSW-NB15_training.csv', index=False)
testing_data.to_csv('UNSW-NB15_testing.csv', index=False)

print("Preprocessing done")

# Print Info
print("Training data shape: ", training_data.shape)
print("Testing data shape: ", testing_data.shape)
print("Total data shape: ", final_data.shape)
print("Total data attack_cat: ")
print(final_data['attack_cat'].value_counts())


