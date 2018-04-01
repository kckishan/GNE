import random
import os

import pickle
from sklearn.preprocessing import scale

from utils import *

# Create Data Loader
# Load expression data file of shape E * N where N is number of genes and E is number of experiments
def load_data(datafile, normalize=True):
    """
    This function loads data set
    :param datafile: 
    :return expression data: 
    """
    # Load data file
    df = pd.read_csv(datafile, sep='\t', header=0)
    df.columns = [int(x[1:]) - 1 for x in df.columns]
    if normalize==True:
        df = pd.DataFrame(scale(df, axis=0))
    t_data = df.T
    return (t_data)

def create_train_test_split(path, adj_matrix, test_size=0.1, validation_size=0.1, save_to_file=True):
    
    print("Creating train test and validation_split")

    # Split link information to train and validation with test split size 
    edgelist = convertAdjMatrixToSortedRankTSV(adj_matrix)
    pos_edges = np.array(edgelist.loc[edgelist.iloc[:, 2] == 1])

    neg_edgelist = np.array(edgelist.loc[edgelist.iloc[:, 2] == 0])
    ind = random.sample(range(len(neg_edgelist)), len(pos_edges))
    neg_edges = pd.DataFrame(np.random.permutation(neg_edgelist))
    neg_edges = neg_edges.iloc[ind, :]

    X_pos, test_edges = train_test_split(pos_edges, test_size=test_size)
    X_neg, test_edges_false = train_test_split(neg_edges, test_size=test_size)


    print(test_size)
    train_edges, val_edges = train_test_split(X_pos, test_size=validation_size)
    train_edges_false, val_edges_false = train_test_split(X_neg, test_size=validation_size)

    dataset = {}
    dataset['train_pos'] = train_edges
    dataset['train_neg'] = train_edges_false
    dataset['val_pos'] = val_edges
    dataset['val_neg'] = val_edges_false
    dataset['test_pos'] = test_edges
    dataset['test_neg'] = test_edges_false

    if save_to_file:
        test_split_file = open( path + "/split_data_"+str(1.0-validation_size)+".pkl", 'wb')
        pickle.dump(dataset, test_split_file)
        test_split_file.close()

    return dataset