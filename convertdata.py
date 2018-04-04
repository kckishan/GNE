import random
import scipy.sparse as sp

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

def create_train_test_split(path, adj, test_size=0.1, validation_size=0.1, save_to_file=True):
    
    print("Creating train test and validation split")
    g = nx.Graph(adj)
    adj = nx.to_scipy_sparse_matrix(g)

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()

    # Split link information to train and validation with test split size
    edgelist = convertAdjMatrixToSortedRankTSV(adj.todense())
    geneids = edgelist.iloc[:,:2]
    col1 = np.array(geneids).min(axis=1).astype(int)
    col2 = np.array(geneids).max(axis=1).astype(int)
    col3 = np.array(edgelist.iloc[:,2])
    data_df = pd.DataFrame()
    data_df['i'] = col1
    data_df['j'] = col2
    data_df['k'] = col3
    data_df = data_df.drop_duplicates()

    pos_edges = data_df.loc[data_df.iloc[:, 2] == 1]
    neg_edgelist = data_df.loc[data_df.iloc[:, 2] == 0]
    ind = random.sample(range(len(neg_edgelist)), pos_edges.shape[0])
    neg_edges = pd.DataFrame(np.random.permutation(neg_edgelist.values))
    neg_edges = neg_edges.iloc[ind, :]

    X_pos, test_edges = train_test_split(pos_edges.values, test_size=test_size)
    X_neg, test_edges_false = train_test_split(neg_edges.values, test_size=test_size)

    train_edges, val_edges = train_test_split(X_pos, test_size=validation_size)
    train_edges_false, val_edges_false = train_test_split(X_neg, test_size=validation_size)


    assert set(map(tuple, test_edges_false)).isdisjoint(set(map(tuple, train_edges)))
    assert set(map(tuple, val_edges_false)).isdisjoint(set(map(tuple, train_edges)))
    assert set(map(tuple, train_edges_false)).isdisjoint(set(map(tuple, train_edges)))

    # assert: test, val, train false edges disjoint
    assert set(map(tuple, test_edges_false)).isdisjoint(set(map(tuple, val_edges_false)))
    assert set(map(tuple, test_edges_false)).isdisjoint(set(map(tuple, train_edges_false)))
    assert set(map(tuple, val_edges_false)).isdisjoint(set(map(tuple, train_edges_false)))

    # assert: test, val, train positive edges disjoint
    assert set(map(tuple, val_edges)).isdisjoint(set(map(tuple, train_edges)))
    assert set(map(tuple, test_edges)).isdisjoint(set(map(tuple, train_edges)))
    assert set(map(tuple, val_edges)).isdisjoint(set(map(tuple, test_edges)))

    dataset = {}
    dataset['train_pos'] = train_edges
    dataset['train_neg'] = train_edges_false
    dataset['val_pos'] = val_edges
    dataset['val_neg'] = val_edges_false
    dataset['test_pos'] = test_edges
    dataset['test_neg'] = test_edges_false

    if save_to_file:
        test_split_file = open( path + "/split_data_"+str(round(1.0-validation_size, 2))+".pkl", 'wb')
        pickle.dump(dataset, test_split_file)
        test_split_file.close()
    return dataset
