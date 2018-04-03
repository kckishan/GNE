import argparse
import time
import pickle
from sklearn.metrics.pairwise import cosine_similarity

import LoadData as data
from evaluation import *
from GNE import GNE
from convertdata import *



parameters = {}
parameters['id_embedding_size'] = 128
parameters['attr_embedding_size'] = 128
parameters['alpha'] = 1
parameters['n_neg_samples'] = 10
parameters['epoch'] = 20
parameters['representation_size'] = 128
parameters['batch_size'] = 256
parameters['learning_rate'] = 0.005
#
# Define the file number associated with organism: yeast or ecoli
organism = 'yeast'

# Define path
path = './data/' + organism +'/'


geneids = pd.read_csv(path + "gene_ids.tsv", sep=" ")
num_genes = geneids.shape[0]
link_file = path + "edgelist_biogrid.txt"
feature_file = path + 'expression_data.tsv'

adj = load_network(link_file, num_genes)

# Perform train-test split
dataset = create_train_test_split(path, adj, test_size=0.1, validation_size=0.1)

train_edges = dataset['train_pos']
train_edges_false = dataset['train_neg']
val_edges = dataset['val_pos']
val_edges_false = dataset['val_neg']
test_edges = dataset['test_pos']
test_edges_false = dataset['test_neg']

# Inspect train/test split
print("Total nodes:", adj.shape[0])
print("Total edges:", np.sum(adj))  # adj is symmetric, so nnz (num non-zero) = 2*num_edges
print("Training edges (positive):", len(train_edges))
print("Training edges (negative):", len(train_edges_false))
print("Validation edges (positive):", len(val_edges))
print("Validation edges (negative):", len(val_edges_false))
print("Test edges (positive):", len(test_edges))
print("Test edges (negative):", len(test_edges_false))

Data = data.LoadData(path, train_links=train_edges, features_file=feature_file)

print("Path: ", path)
print("Total number of nodes: ", Data.id_N)
print("Total number of attributes: ", Data.attr_M)
print("Total training links: ", len(Data.links))
print("Total epoch: ", parameters['epoch'])
print('Dimension of structural Embedding (d):', parameters['id_embedding_size'])
print('Dimension of attribute Embedding (d):', parameters['attr_embedding_size'])
print('Dimension of final representation (d):', parameters['representation_size'])
#
# Create validation edges and labels
validation_edges = np.concatenate([val_edges, val_edges_false])
val_edge_labels = np.concatenate([np.ones(len(val_edges)), np.zeros(len(val_edges_false))])

test_edges_data = np.concatenate([test_edges, test_edges_false])
test_edge_labels = np.concatenate([np.ones(len(test_edges)), np.zeros(len(test_edges_false))])

for alpha in  [0, 0.2, 0.4, 0.6, 0.8, 1]:
    parameters['alpha'] = alpha
    model = GNE(path, Data, 2018, parameters)
    embeddings = model.train(validation_edges, val_edge_labels)

        # Train-set edge embeddings
    pos_train_edge_embs = get_edge_embeddings(embeddings, train_edges)

    neg_train_edge_embs = get_edge_embeddings(embeddings, train_edges_false)
    train_edge_embs = np.concatenate([pos_train_edge_embs, neg_train_edge_embs])

    # Create train-set edge labels: 1 = real edge, 0 = false edge
    train_edge_labels = np.concatenate([np.ones(len(train_edges)), np.zeros(len(train_edges_false))])

    index = np.random.permutation([i for i in range(len(train_edge_labels))])
    train_data = train_edge_embs[index, :]
    train_labels = train_edge_labels[index]

    # Test-set edge embeddings, labels
    pos_test_edge_embs = get_edge_embeddings(embeddings, test_edges)
    neg_test_edge_embs = get_edge_embeddings(embeddings, test_edges_false)
    test_edge_embs = np.concatenate([pos_test_edge_embs, neg_test_edge_embs])

    # Create val-set edge labels: 1 = real edge, 0 = false edge

    # Train logistic regression classifier on train-set edge embeddings
    from sklearn.linear_model import LogisticRegression

    edge_classifier = LogisticRegression(random_state=0)
    edge_classifier.fit(train_data, train_labels)

    index = np.random.permutation([i for i in range(len(test_edge_labels))])
    test_data = test_edge_embs[index, :]
    test_labels = test_edge_labels[index]

    test_preds = edge_classifier.predict_proba(test_data)[:, 1]
    test_roc = roc_auc_score(test_labels, test_preds)
    test_ap = average_precision_score(test_labels, test_preds)

    # link prediction test
    # distance_matrix = -1 * euclidean_distances(embeddings, embeddings)
    # test_roc, test_ap = evaluate_ROC_from_matrix(test_edges_data, test_edge_labels, distance_matrix)
    print("Alpha :", str(alpha))
    print('GNE Test ROC score: ', str(test_roc))
    print('GNE Test AP score: ', str(test_ap))

    embeddings_file = open(path + "embeddings_trainsize_" + str(train_size) + "_alpha_"+str(alpha)+".pkl", 'wb')
    pickle.dump(embeddings, embeddings_file)
    embeddings_file.close()
