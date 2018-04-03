#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 10:26:11 2017

@author: kishan_kc
"""
import csv
from random import randint

import numpy as np
from dreamtools import D5C4
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import networkx as nx
from numba import jit
import scipy.io as sio
from scipy.sparse import coo_matrix

def plot_correlation_matrix(corr):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(corr, interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Gene Expression Correlation')
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=np.arange(-1.0, 1.0, 0.05))
    plt.show()


def transformGraphToAdj(graph):
    n = graph.number_of_nodes()
    adj = np.zeros((n, n))

    for (src, dst, w) in graph.edges(data="weight", default=1):
        adj[src, dst] = w

    return w


def saveGraphToEdgeListTxt(graph, file_name):
    with open(file_name, 'w') as f:
        f.write('#d\n' % graph.number_of_nodes())
        f.write('#d\n' % graph.number_of_edges())
        for i, j, w in graph.edges(data='weight', default=1):
            f.write('#d #d #f\n' % (i, j, w))


def saveGraphToEdgeListTxtn2v(graph, file_name):
    with open(file_name, 'w') as f:
        for i, j, w in graph.edges(data='weight', default=1):
            f.write('#d #d #f\n' % (i, j, w))


def splitGraphToTrainTest(graph, train_ratio, is_undirected=True):
    train_graph = graph.copy()
    test_graph = graph.copy()
    node_num = graph.number_of_nodes()
    for (st, ed, w) in graph.edges(data='weight', default=1):
        if (is_undirected and st >= ed):
            continue
        if (np.random.uniform() <= train_ratio):
            test_graph.remove_edge(st, ed)
            if (is_undirected):
                test_graph.remove_edge(ed, st)
        else:
            train_graph.remove_edge(st, ed)
            if (is_undirected):
                train_graph.remove_edge(ed, st)
    return (train_graph, test_graph)


def convertEdgeListToAdj(adj, threshold=0.0, is_undirected=True, edge_pairs=None):
    result = []
    node_num = adj.shape[0]
    if edge_pairs:
        for (st, ed) in edge_pairs:
            if adj[st, ed] >= threshold:
                result.append((st, ed, adj[st, ed]))
    else:
        for i in range(node_num):
            for j in range(node_num):
                if (j == i):
                    continue
                if (is_undirected and i >= j):
                    continue
                if adj[i, j] > threshold:
                    result.append((i, j, adj[i, j]))
    return result


def splitDataToTrainTest(data, selection=None):
    n = np.sum(data[:, -1]) * 2
    if selection:
        X = data[0:int(n), 0:-1]
        y = data[0:int(n), -1]
    else:
        X = data[:, 0:-1]
        y = data[:, -1]

    # scaler = StandardScaler()
    # X_train_scaler = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y,   stratify=y,  test_size=0.20, random_state=42)
    return X_train, X_test, y_train, y_test


def balancedErrorRate(actual, output):
    cm = confusion_matrix(actual, output)
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]
    return 0.5 * ((float(FN) / float(TP + FN)) + (float(FP) / float(FP + TN)))

def saveToFile(filename, data):
    fileHandler = open(filename,'w')
    fileHandler.write(data)
    fileHandler.close()

# Create Data Loader
def load_data(datafile):
    """
    This function loads data set
    :param datafile:
    :return expression data:
    """
    # Load data file
    df = pd.read_csv(datafile, sep='\t', header= 0)
    t_data = df.T
    return(t_data)


def loadEmbedding(file_name):
    with open(file_name, 'r') as f:
        n, d = f.readline().strip().split()
        X = np.zeros((int(n), int(d)))
        for line in f:
            emb = line.strip().split()
            emb_fl = [float(emb_i) for emb_i in emb[1:]]
            X[int(emb[0]), :] = emb_fl
    return X

def load_gold_standard(file_name, sep="\t"):
    df = pd.read_csv(file_name, sep=sep, header=None)
    # Load gold standard relation file
    df[0] = df[0].apply(lambda x: x.replace('g', '').replace('G', ''))
    df[1] = df[1].apply(lambda x: x.replace('g', '').replace('G', ''))
    df = df.astype(float)  # imoprtant for later to check for equality
    df[0] = df[0].apply(lambda x: x - 1)
    df[1] = df[1].apply(lambda x: x - 1)
    df = df.astype(float)  # imoprtant for later to check for equality
    return df

# Create Graph based on Correlation Matrix

def generateMultipleGraphs(correlation_matrix, num_of_thresholds):
    """
    :param correlation_matrix:
    :param num_of_thresholds:
    :param range:
    :return: Multiple adjacency matrix based on thresholds
    """
    mat = np.abs(correlation_matrix).copy()
    mat = mat - np.diag(np.ones(mat.shape[0]))
    final = nx.Graph()
    for i in range(num_of_thresholds):
        threshold = randint(20, 90)/100
        graph = generateGraphs(mat, threshold)
        G = nx.Graph(graph)
        final = nx.compose(final, G)
    return final

def generateGraphs(correlation_matrix, threshold):
    """
    :param correlation_matrix:
    :param threshold:
    :return: Adjacency matrix based on correlation threshold
    """
    threshold_matrix = np.abs(correlation_matrix).copy()
    threshold_matrix = threshold_matrix - np.diag(np.ones(threshold_matrix.shape[0]))
    threshold_matrix[np.abs(correlation_matrix)<threshold] = 0
    return (threshold_matrix)


# def biogrid_goldStandard(gold_standard, gene_lists):
def biogrid_goldStandard():
    gene_list = pd.read_csv("../data/net4_gene_ids.tsv")
    file_name = '../data/BIOGRID-ALL-3.4.153.tab2.txt'
    gold_standard_data = pd.read_csv(file_name, sep="\t")
    yeast_data = gold_standard_data[(gold_standard_data['Organism Interactor A'] == 559292) & (gold_standard_data['Organism Interactor B'] == 559292)]
    yeast_data = yeast_data[['Systematic Name Interactor A', 'Systematic Name Interactor B']]
    yeast_data.to_csv("../data/edgelist_biogrid.csv", index=False, header=False)

def adj_to_list(A, output_filename, delimiter):
    '''Takes the adjacency matrix on file input_filename into a list of edges and saves it into output_filename'''
    List = [('Source', 'Target', 'Weight')]
    for source in A.index.values:
        for target in A.index.values:
            List.append((target, source, A[source][target]))
    with open(output_filename, "wb") as f:
        writer = csv.writer(f)
        writer.writerows(List)
    return List


# Implement DeepWalk, SDNE, GCN, GVAE

# Implement Hadamard, Average, L1, L2 approach to combine two nodes
def average(u, v):
    return np.add(u, v)/2

def hadamard(u, v):
    return np.multiply(u, v)

def weighted_L1(u, v):
    return np.abs(u - v)

def weighted_L2(u, v):
    return np.square(u - v)

@jit
# Create Feature for Classification using embedding
def createFeaturesFromEmbedding(emb, gold_standard, method="concatenate"):
    if method =="concatenate":
        columns = 2*int(emb.shape[1])
    else:
        columns = int(emb.shape[1])

    data = np.zeros((gold_standard.shape[0],columns))
    labels = gold_standard[:, 2]

    for i in range(gold_standard.shape[0]):
        if method=="concatenate":
            data[i] = np.concatenate([emb[int(gold_standard[i, 0]), :], emb[int(gold_standard[i, 1]), :]])
        elif method == "average":
            data[i] = average(emb[int(gold_standard[i, 0]), :], emb[int(gold_standard[i, 1]), :])
        elif method == "hadamard":
            data[i] = hadamard(emb[int(gold_standard[i, 0]), :], emb[int(gold_standard[i, 1]), :])
        elif method == "weighted_L1":
            data[i] = weighted_L1(emb[int(gold_standard[i, 0]), :], emb[int(gold_standard[i, 1]), :])
        elif method == "weighted_L2":
            data[i] = weighted_L2(emb[int(gold_standard[i, 0]), :], emb[int(gold_standard[i, 1]), :])

    return data, labels

def convertAdjMatrixToSortedRankTSV(inputFile=None, outputFilename=None, desc=True):
    tbl = inputFile

    rownames = range(tbl.shape[0])
    # First column -> repeat the predictors
    firstCol = np.repeat(rownames, tbl.shape[1]).reshape((tbl.shape[0]*tbl.shape[1], 1))
    # Second column -> repeat the targets
    secondCol = []

    x = np.array([i for i in range(tbl.shape[0])])
    secondCol = np.tile(x, len(range(tbl.shape[1])))
    # for i in range(tbl.shape[1]):
    #     # print(i)
    #     secondCol = np.append(secondCol, range(tbl.shape[0]))
    # print(len(secondCol))

    secondCol = secondCol.reshape((tbl.shape[0]*tbl.shape[1], 1))
    thirdCol = np.matrix.flatten(np.matrix(tbl)).reshape((tbl.shape[0]*tbl.shape[1], 1))
    thirdCol = np.nan_to_num(thirdCol)
    # Gets the indices from a desc sort on the adjancy measures

    # Glue everything together
    result = np.column_stack((firstCol, secondCol, thirdCol))
    # Convert to dataframe
    result = pd.DataFrame(result)
    result.columns = ['c1','c2', 'c3']

    result = pd.DataFrame(result[result['c1']!=result['c2']])
    # Sort it using the indices obtained before
    result =  result.sort_values(['c3', 'c1', 'c2'], ascending=[0, 1, 1])
    result[['c1', 'c2']] = result[['c1', 'c2']].astype(int)
    # print("Write to file if filename is given")
    # result.to_csv(outputFilename, header=False, columns=None, index=False )
    # else write to function output
    return (result)


def convertSortedRankTSVToAdjMatrix (input=None, nodes=None, undirected=True):
    print("Converting to Adjacency matrix")
    tbl = pd.DataFrame(input).drop_duplicates()
    tbl.columns = ['c1', 'c2', 'c3']
    tbl = tbl[tbl['c3']==1]

    tbl = tbl.sort_values(['c2'], ascending=1)
    tbl[['c1', 'c2', 'c3']] = tbl[['c1', 'c2', 'c3']].astype(int)
    # Pre allocate return matrix
    m = np.zeros((nodes, nodes))
    # Get the duplicates
    dups = tbl['c2'].duplicated()

    # # Get the startIndices of another column
    startIndices = list(np.where(dups== False)[0])

    for i in range(len(startIndices)-1):
        # print(i)
        colIndex = tbl.iloc[startIndices[i], 1]
        if startIndices[i]==(startIndices[i + 1] - 1):
            rowIndexes = tbl.iloc[startIndices[i], 0]
            valuesToAdd = tbl.iloc[startIndices[i], 2]
        else:
            rowIndexes = tbl.iloc[startIndices[i]:(startIndices[i + 1]), 0].values
            valuesToAdd = tbl.iloc[startIndices[i]:(startIndices[i + 1] ), 2].values
        m[rowIndexes, colIndex] = valuesToAdd
        if undirected:
            m[colIndex, rowIndexes] = valuesToAdd


    colIndex = tbl.iloc[startIndices[len(startIndices)-1], 1]
    rowIndexes = tbl.iloc[startIndices[len(startIndices)-1]:len(tbl.iloc[:, 1]), 0]
    valuesToAdd = tbl.iloc[startIndices[len(startIndices)-1]:len(tbl.iloc[:, 1]), 2]

    m[rowIndexes, colIndex] = valuesToAdd
    if undirected:
        m[colIndex, rowIndexes] = valuesToAdd

    m = pd.DataFrame(m)
    # m.to_csv(outputFilename, header=False, columns=None, index=False )
    # else write to function output
    return (m)


# copy and paste from D5C3
def score_prediction(prediction, gold_edges):
    """
    :param filename:
    :param tag:
    :return:
    """

    challenge = D5C4()

    # DISCOVERY
    # In principle we could resuse ROCDiscovery class but
    # here the pvaluse were also computed. let us do it here for now

    merged = pd.merge(gold_edges, prediction, how='inner', on=[0, 1])

    TPF = len(merged)

    gold_edges = gold_edges.astype(int)

    # unique species should be 1000
    N = len(set(gold_edges[0]).union(gold_edges[1]))
    # positive
    print('Scanning gold standard')
    # should be 4012, 274380 and 178 on template
    G = challenge._get_G(gold_edges, prediction)

    # get back the sparse version for later
    # keep it local to speed up import
    import scipy.sparse
    H = scipy.sparse.csr_matrix(G>0)

    Pos = sum(sum(G > 0))
    Neg = sum(sum(G < 0))
    Ntot = Pos + Neg

    print("\n Cleaning up the prediction that are not in the GS")
    newpred = challenge._remove_edges_not_in_gs(prediction, G)
    L = len(newpred)


    discovery = np.zeros(L)
    X = [tuple(map(int, np.nan_to_num(x))) for x in newpred[[0,1]].values-1]

    discovery = [H[x] for x in X]
    TPL = sum(discovery)

    discovery = np.array([int(x) for x in discovery])

    if L < Ntot:
        p = (Pos - TPL) / float(Ntot - L)
    else:
        p = 0

    random_positive_discovery = [p] * (Ntot - L)
    random_negative_discovery = [1-p] * (Ntot - L)

    # append discovery + random using lists
    positive_discovery = np.array(list(discovery) + random_positive_discovery)
    negative_discovery = np.array(list(1-discovery) + random_negative_discovery)

    #  true positives (false positives) at depth k
    TPk = np.cumsum(positive_discovery)
    FPk = np.cumsum(negative_discovery)

    #  metrics
    TPR = TPk / float(Pos)
    FPR = FPk / float(Neg)
    REC = TPR  # same thing
    PREC = TPk / range(1,Ntot+1)

    from dreamtools.core.rocs import ROCBase
    roc = ROCBase()
    auroc = roc.compute_auc(roc={'tpr':TPR, 'fpr':FPR})
    aupr = roc.compute_aupr(roc={'precision':PREC, 'recall':REC})

    # normalise by max possible value
    aupr /= (1.-1./Pos)

    results = {'auroc':auroc, 'aupr':aupr}
    return results

@jit
def read_test_link(testlinkfile):
    X_test = []
    f = open(testlinkfile)
    line = f.readline()
    while line:
        line = line.strip().split(' ')
        X_test.append([int(line[0])  ,  int(line[1])  ,  int(line[2]) ])
        line = f.readline()
    f.close()
    print("test link number:", len(X_test))
    return X_test


def load_network(filename, num_genes):
    print ("### Loading [%s]..." % (filename))
    i, j, val = np.loadtxt(filename).T
    A = coo_matrix((val, (i, j)), shape=(num_genes, num_genes))
    A = A.todense()
    A = np.squeeze(np.asarray(A))
    A = A - np.diag(np.diag(A))
    return A