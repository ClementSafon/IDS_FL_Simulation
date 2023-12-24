import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import re
from numpy.linalg import inv
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
import seaborn as sns
from scipy.linalg import pinv


def count_true_false_repartition(n: int):

    def count_true_false(array):
        true_count = np.count_nonzero(array == True)
        false_count = np.count_nonzero(array == False)
        return true_count, false_count

    # Load the npz file
    files_path_prefix = 'data_party'
    files_path_suffix = '.npz'
    for i in range(n):
        file_path = files_path_prefix + str(i) + files_path_suffix
        print("File path:", file_path)
        data = np.load(file_path)

        # Extract y_train and y_test arrays
        y_train = data['y_train']
        y_test = data['y_test']

        # Count true and false values in y_train
        true_count_train, false_count_train = count_true_false(y_train[:,0])
        print("     For y_train:")
        print("     True count:", true_count_train)
        print("     False count:", false_count_train)

        # Count true and false values in y_test
        true_count_test, false_count_test = count_true_false(y_test[:,0])
        print("\n       For y_test:")
        print("     True count:", true_count_test)
        print("     False count:", false_count_test)
        print("\n")


def stat_ip():
    csv = pd.read_csv("dataset/UNSW-NB15/NUSW-NB15_features.csv", encoding='latin-1')
    all_c_names = []
    for i in range(len(csv)):
        all_c_names.append(csv['Name'][i])

    with open("dataset/UNSW-NB15/a part of training and testing set/UNSW_NB15_training-set.csv","r") as f:
        useful_c_names = f.readline()[1:-1].split(",")

    files_path = ["dataset/UNSW-NB15/UNSW-NB15_1.csv","dataset/UNSW-NB15/UNSW-NB15_2.csv","dataset/UNSW-NB15/UNSW-NB15_3.csv","dataset/UNSW-NB15/UNSW-NB15_4.csv"]

    ip_list = set()

    for file_path in files_path:

        data = pd.read_csv(file_path, names=all_c_names)

        ip_dst = set(data['dstip'].unique())
        ip_src = set(data['srcip'].unique())

        ip_list = ip_list.union(ip_dst)
        ip_list = ip_list.union(ip_src)
    
    ip_list = list(ip_list)
    ip_list.sort()

    for ip in ip_list:
        print(ip)

def create_datasets_with_ip():
    csv = pd.read_csv("dataset/UNSW-NB15/NUSW-NB15_features.csv", encoding='latin-1')
    all_c_names = []
    for i in range(len(csv)):
        all_c_names.append(csv['Name'][i])

    with open("dataset/UNSW-NB15/a part of training and testing set/UNSW_NB15_training-set.csv","r") as f:
        useful_c_names = f.readline()[1:-1].split(",")

    files_path = ["dataset/UNSW-NB15/UNSW-NB15_1.csv","dataset/UNSW-NB15/UNSW-NB15_2.csv","dataset/UNSW-NB15/UNSW-NB15_3.csv","dataset/UNSW-NB15/UNSW-NB15_4.csv"]

    ip_list = set()

    for file_path in files_path:

        data = pd.read_csv(file_path, names=all_c_names)

        ip_dst = set(data['dstip'].unique())
        ip_src = set(data['srcip'].unique())

        ip_list = ip_list.union(ip_dst)
        ip_list = ip_list.union(ip_src)
    
    ip_list = list(ip_list)
    ip_list.sort()

    for ip in ip_list:
        print(ip)

def tmp():
    # take x y m data from .npz file
    data = np.load("data_client_no_analysis/party0.npz")
    x_train = data['x_train']
    y_train = data['y_train']

    for i in range(y_train.shape[1]):
        print(np.unique(y_train[:,i]))

def count_labels():
    features = pd.read_csv("dataset/UNSW-NB15/NUSW-NB15_features.csv", encoding='latin-1')
    c_names = features['Name'].to_list()
    c_types = features['Type '].to_list()

    # replace "nominal" by "string"
    for i in range(len(c_types)):
        if c_types[i] == "nominal":
            c_types[i] = "string"
    # replace "integer" or "Integer" by "int64"
    for i in range(len(c_types)):
        if c_types[i] == "integer" or c_types[i] == "Integer":
            c_types[i] = "string"
    # replace "Float" by "float64"
    for i in range(len(c_types)):
        if c_types[i] == "Float" or c_types[i] == "Timestamp":
            c_types[i] = "float"
    # replace "binary" by "bool"
    for i in range(len(c_types)):
        if c_types[i] == "binary" or c_types[i] == "Binary":
            c_types[i] = "string"

    sizes = {}

    for i in tqdm(range(1,5)):
        content = pd.read_csv("dataset/UNSW-NB15/UNSW-NB15_"+str(i)+".csv", names=c_names, dtype=dict(zip(c_names, c_types)))
        labels = content['attack_cat'].unique()

        for i in range(len(labels)):
            if labels[i] is not pd.NA:
                labels[i] = labels[i].strip()
                if labels[i] not in sizes.keys():
                    sizes[labels[i]] = 0
            else:
                labels[i] = "Normal"
                if "Normal" not in sizes.keys():
                    sizes["Normal"] = 0
            

        for label in labels:
            if label == "Normal":
                sizes[label] += len(content[content['Label'] == "0"])
            else:
                content['attack_cat'] = content['attack_cat'].apply(lambda x: x.strip() if isinstance(x, str) else x)
                sizes[label] += len(content[content['attack_cat']== label])
   
    total = 0
    for key in sizes.keys():
        total += sizes[key]

    print("{")
    for key in sizes.keys():
        print("    \""+key+"\": "+str(sizes[key])+" ("+str(round(100*sizes[key]/total, 2))+"%),")

    print("}")

def count_labels2():
    data = pd.read_csv("dataset/UNSW-NB15/a part of training and testing set/UNSW_NB15_training-set.csv", encoding='latin-1')
    labels = data['attack_cat'].unique()

    sizes = {}

    for i in range(len(labels)):
        labels[i] = labels[i].strip()
        if labels[i] not in sizes.keys():
            sizes[labels[i]] = 0
    
    for label in labels:
        sizes[label] = len(data[data['attack_cat']== label])

    total = 0
    for key in sizes.keys():
        total += sizes[key]

    print("{")
    for key in sizes.keys():
        print("    \""+key+"\": "+str(sizes[key])+" ("+str(round(100*sizes[key]/total, 2))+"%),")

    print("}")
    data = pd.read_csv("dataset/UNSW-NB15/a part of training and testing set/UNSW_NB15_testing-set.csv", encoding='latin-1')
    labels = data['attack_cat'].unique()

    sizes = {}

    for i in range(len(labels)):
        labels[i] = labels[i].strip()
        if labels[i] not in sizes.keys():
            sizes[labels[i]] = 0
    
    for label in labels:
        sizes[label] = len(data[data['attack_cat']== label])

    total = 0
    for key in sizes.keys():
        total += sizes[key]

    print("{")
    for key in sizes.keys():
        print("    \""+key+"\": "+str(sizes[key])+" ("+str(round(100*sizes[key]/total, 2))+"%),")

    print("}")

def display_features():
    features = pd.read_csv("dataset/UNSW-NB15/NUSW-NB15_features.csv", encoding='latin-1')
    c_names = features['Name'].to_list()

    
    usefull_features = pd.read_csv("dataset/UNSW-NB15/a part of training and testing set/UNSW_NB15_training-set.csv", encoding='latin-1')
    usefull_features = usefull_features.columns.to_list()

    for i in range(len(usefull_features)):
        usefull_features[i] = usefull_features[i].strip().lower()
    for i in range(len(c_names)):
        c_names[i] = c_names[i].strip().lower()

    print("{")
    for i in range(len(c_names)):
        if c_names[i] in usefull_features:
            print("     " + c_names[i]+", "+c_names[i]+",")
        else:
            print("     "+c_names[i]+", --- ,")
    print("}")

def plot_figs():
    # Open the markdown file and read its content
    with open('notes.md', 'r') as file:
        data = file.read()

    # Use regular expressions to find the data blocks that look like dictionaries
    pattern = re.compile(r'\{(.*?)(\})', re.DOTALL)
    matches = pattern.findall(data)

    matches.remove(matches[1])

    # Parse these blocks to create Python dictionaries
    dictionaries = []
    for match in matches:
        dict_data = match[0].split('\n')
        dict_data = [item.strip().split(':') for item in dict_data if item.strip()]
        dictionary = {item[0].strip('\"'): int(item[1].split()[0]) for item in dict_data}
        dictionaries.append(dictionary)
    


    # Use matplotlib to plot the data from these dictionaries
    for i, dictionary in enumerate(dictionaries):
        keys = sorted(dictionary.keys())
        values = [dictionary[key] for key in keys]
        if i == 0:
            plt.figure(i)
            plt.bar(keys, values)
            plt.xticks(rotation=90)
            plt.title(f'UNSW-NB15 Data Distribution')
            plt.tight_layout()
        elif i == 1:
            plt.figure(i)
            plt.bar(keys, values, label='Training set')
            plt.xticks(rotation=90)
            plt.title('UNSW-NB15 (example) Data Distribution')
            plt.tight_layout()
            plt.legend()
        elif i == 2:
            plt.figure(i-1)
            prev_values = [dictionaries[i-1][key] for key in keys]
            plt.bar(keys, values, bottom=prev_values, label='Testing set')
            plt.xticks(rotation=90)
            plt.title('UNSW-NB15 (example) Data Distribution')
            plt.tight_layout()
            plt.legend()
    plt.show()


def mahalanobis_distance():
    data = pd.read_csv("dataset/UNSW-NB15/a part of training and testing set/UNSW_NB15_testing-set.csv", encoding='latin-1')
    data = data.drop(['ï»¿id', 'label'], axis=1)
    
    attack_cat_encoded = pd.get_dummies(data['attack_cat'])

    data_encoded = pd.concat([data.drop('attack_cat', axis=1), attack_cat_encoded], axis=1)
    data_encoded = pd.get_dummies(data_encoded)

    ### Normalization
    # scaler = MinMaxScaler()
    # scaler.fit(data_encoded)
    # data_encoded[data_encoded.columns] = scaler.transform(data_encoded)

    scaler = QuantileTransformer()
    data_encoded_norm = scaler.fit_transform(data_encoded)
    data_encoded = pd.DataFrame(data_encoded_norm, columns=data_encoded.columns)


    # Calculate centroids for each class
    centroids = {}
    for column in attack_cat_encoded.columns:
        encoded_rows = data_encoded[data_encoded[column] == True].copy()
        encoded_rows = encoded_rows.drop(attack_cat_encoded.columns, axis=1)
        centroids[column] = encoded_rows.mean()
    centroids = pd.DataFrame(centroids).T

    # Get the encoded data for each class
    data_encoded_by_class = {}
    for column in attack_cat_encoded.columns:
        data_encoded_by_class[column] = data_encoded[data_encoded[column] == True]
        data_encoded_by_class[column] = data_encoded_by_class[column].drop(attack_cat_encoded.columns, axis=1)


    print("Centroids:")
    print(centroids)
    print("")
    

    def mahalanobis(x, data):
        x = np.array(x).reshape(1, -1)
        x_minus_mu = x - np.mean(data)
        try:
            cov = np.cov(data.T)
            inv_covmat = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            print("Singular matrix - Computing pseudo-cov inv")
            cov = np.cov(data.T) + 1e-8*np.eye(data.shape[1])  
            inv_covmat = np.linalg.inv(cov)
        # cov = np.cov(data.T)
        # inv_covmat = pinv(cov)
        left_term = np.dot(x_minus_mu, inv_covmat)
        mahal = np.dot(left_term, x_minus_mu.T)
        return mahal.diagonal()

    distances = pd.DataFrame(index=centroids.index, columns=centroids.index)

    for index, i in enumerate(centroids.index):
        for j in centroids.index[index+1:]:
            distances.loc[i, j] = mahalanobis(centroids.loc[i], data_encoded_by_class[j])

    print(distances)

    distances = distances.T.astype(float)
    distances = distances / distances.max().max()

    sns.heatmap(distances, cmap='viridis')
    plt.title('Mahalanobis distances between centroids')
    plt.show()

if __name__ == "__main__":
    # count_true_false_repartition(3)

    # stat_ip()

    # tmp()

    # count_labels()
    # count_labels2()

    # display_features()

    # plot_figs()

    mahalanobis_distance()