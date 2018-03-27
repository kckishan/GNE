import random
import os
from sklearn.preprocessing import scale

from utils import *

# Create Data Loader
# Load expression data file of shape E * N where N is number of genes and E is number of experiments
def load_data(datafile):
    """
    This function loads data set
    :param datafile: 
    :return expression data: 
    """
    # Load data file
    df = pd.read_csv(datafile, sep='\t', header=0)
    df.columns = [int(x[1:]) - 1 for x in df.columns]
    df = pd.DataFrame(scale(df, axis=0))
    t_data = df.T
    return (t_data)

# Function to check if file exists and remove it
def remove_file(file):
    if os.path.isfile(file) == True:
        os.remove(file)

# Create test file from gold standard with defined test size
def create_test_file(path, link_file, train_file, test_file, test_size=0.1):
    remove_file(test_file)
    remove_file(train_file)

    edgelist = pd.read_csv(link_file, sep=" ", header=None)

    # Read the saved negative interactions examples 
    neg_adj = pd.read_csv(path + "neg_sample.txt", header=None, sep=",")
    neg_adj.columns = ['A', 'B', 'C']

    # Split link information to train and test with test split size 
    edgelist = edgelist.sample(frac=1).reset_index(drop=True)
    x_train, x_test = train_test_split(edgelist, test_size=test_size)
    neg_adj_0 = neg_adj

    # Sample the negative samples as same number as positive instances in test data
    n = x_test.shape[0]
    ind = random.sample(range(len(neg_adj_0)), n)

    # randomize to select random sample
    neg_adj_0 = pd.DataFrame(np.random.permutation(neg_adj_0))
    X_test_0 = neg_adj_0.iloc[ind, :]

    # Stack positive and negative interactions
    X_test = pd.DataFrame(np.vstack((x_test, X_test_0))).astype(int)

    # Save train and test file
    X_test.to_csv(test_file, header=None, sep=' ', index=False, mode='a')
    x_train.to_csv(train_file, header=None, sep=' ', index=False, mode='a')


def convertdata(path, datafile, original_train_file, train_file, validation_file, test_size=0.1):
    
    print("converting data from " + path)
    remove_file(train_file)
    remove_file(validation_file)
    remove_file(path + 'data_standard.txt')

    # Normalize expression data
    data = load_data(datafile)
    data.to_csv(path + 'data_standard.txt', header=None, sep=' ', mode='a')

    # Read edge information
    edgelist = pd.read_csv(path + original_train_file, sep=" ", header=None)

    # Read negative samples
    neg_adj = pd.read_csv(path + "neg_sample.txt", header = None, sep=",")
    neg_adj.columns = ['A', 'B', 'C']

    # Split link information to train and validation with test split size 
    edgelist = edgelist.sample(frac=1).reset_index(drop=True)
    x_train, x_validation = train_test_split(edgelist, test_size=test_size)
    neg_adj_0 = neg_adj

    # randomize to select random sample
    n = x_validation.shape[0]
    ind = random.sample(range(len(neg_adj_0)), n)
    neg_adj_0 = pd.DataFrame(np.random.permutation(neg_adj_0))
    x_validation_0 = neg_adj_0.iloc[ind, :]

    # Stack positive and negative interactions
    X_validation = pd.DataFrame(np.vstack((x_validation, x_validation_0))).astype(int)

    # Save train and test file
    X_validation.to_csv(validation_file, header=None, sep=' ', index=False, mode='a')
    x_train.to_csv(train_file, header=None, sep=' ', index=False, mode='a')
