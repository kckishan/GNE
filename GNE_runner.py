import argparse
import time
import pickle
from sklearn.metrics.pairwise import cosine_similarity


import LoadData as data
from evaluation import *
from GNE import GNE
from convertdata import *

def parse_args():
    parser = argparse.ArgumentParser(description="Run GNE.")
    parser.add_argument('--organism',    nargs='?',                 default='ecoli',    help='Organism to run model for')
    parser.add_argument('--id_dim',                 type=int,       default=128,        help='Dimension for structure embedding')
    parser.add_argument('--attr_dim',               type=int,       default=128,        help='Dimension for attribute embedding')
    parser.add_argument('--epoch',                  type=int,       default=20,         help='Number of epochs for training GNE')
    parser.add_argument('--n_neg_samples',          type=int,       default=10,         help='Number of negative samples for negative sampling optimization')
    parser.add_argument('--batch_size',             type=int,       default=128,        help='Batch size for training GNE')
    parser.add_argument('--representation_size',    type=int,       default=128,        help='Dimension of representation vector')
    parser.add_argument('--learning_rate',          type=float,     default=0.002,      help='Learning rate for optimization')
    parser.add_argument('--alpha',                  type=float,     default=1.0,        help='Relative importance of attribute')
    return parser.parse_args()


if __name__ == '__main__':
    # # parse the arguments
    args = parse_args()
    #
    # Define the file number associated with organism: yeast or ecoli
    organism = args.organism

    # Define path
    path = './data/' + organism +'/'

    if args.organism == "ecoli":
        organism_id = 3
    elif args.organism == "yeast":
        organism_id = 4


    geneids = pd.read_csv(path + "gene_ids.tsv", sep=" ")
    num_genes = geneids.shape[0]
    link_file = path + "edgelist_biogrid.txt"
    feature_file = './data/ecoli/expression_data.tsv'

    adj = load_network(link_file, num_genes)

    # Perform train-test split
    # dataset = create_train_test_split(path, adj, test_size=0.1, validation_size=0.1)
    train_size = 0.9
    # for train_size in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    test_split_file = open(path + "split_data_" + str(train_size) + ".pkl", 'rb')
    dataset = pickle.load(test_split_file)
    test_split_file.close()

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
    #
    Data = data.LoadData(path, train_links=train_edges, features_file=feature_file)
    #
    #
    print("Path: ", path)
    print("Total number of nodes: ", Data.id_N)
    print("Total number of attributes: ", Data.attr_M)
    print("Total training links: ", len(Data.links))
    print("Total epoch: ", args.epoch)
    print('Dimension of Structural Embedding (d):', args.id_dim)
    print('Dimension of Attribute Embedding (d):', args.attr_dim)
    print('Dimension of final representation (d):', args.attr_dim)
    #
    # Create validation edges and labels
    validation_edges = np.concatenate([val_edges, val_edges_false])
    val_edge_labels = np.concatenate([np.ones(len(val_edges)), np.zeros(len(val_edges_false))])

    for alpha in  [0, 0.2, 0.4, 0.6, 0.8, 1]:
        model = GNE(path, Data, alpha=alpha)
        embeddings = model.train(validation_edges, val_edge_labels)

        # Train-set edge embeddings
        pos_train_edge_embs = get_edge_embeddings(embeddings, train_edges)
        neg_train_edge_embs = get_edge_embeddings(embeddings, train_edges_false)
        train_edge_embs = np.concatenate([pos_train_edge_embs, neg_train_edge_embs])

        # Create train-set edge labels: 1 = real edge, 0 = false edge
        train_edge_labels = np.concatenate([np.ones(len(train_edges)), np.zeros(len(train_edges_false))])

        # Test-set edge embeddings, labels
        pos_test_edge_embs = get_edge_embeddings(embeddings, test_edges)
        neg_test_edge_embs = get_edge_embeddings(embeddings, test_edges_false)
        test_edge_embs = np.concatenate([pos_test_edge_embs, neg_test_edge_embs])

        # Create val-set edge labels: 1 = real edge, 0 = false edge
        test_edge_labels = np.concatenate([np.ones(len(test_edges)), np.zeros(len(test_edges_false))])

        # Train logistic regression classifier on train-set edge embeddings
        from sklearn.linear_model import LogisticRegression

        edge_classifier = LogisticRegression(random_state=0)
        edge_classifier.fit(train_edge_embs, train_edge_labels)

        test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]
        test_roc = roc_auc_score(test_edge_labels, test_preds)
        test_ap = average_precision_score(test_edge_labels, test_preds)

        print("Alpha :", str(alpha))
        print('GNE Test ROC score: ', str(test_roc))
        print('GNE Test AP score: ', str(test_ap))