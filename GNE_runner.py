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


def run_GNE(path, data, args):
    t1      = time.time()

    # Execute the model with parsed arguments
    model   = GNE(  path, data, id_embedding_size=args.id_dim, attr_embedding_size=args.attr_dim, batch_size=args.batch_size, alpha=args.alpha, epoch = args.epoch, representation_size=args.representation_size, learning_rate=args.learning_rate)
    embeddings = model.train( )

    t2 = time.time()
    print("Time taken to execute GNE: " + str(t2 - t1))

    return embeddings

if __name__ == '__main__':

    # parse the arguments
    args = parse_args()

    # Define the file number associated with organism: yeast or ecoli
    organism = args.organism

    # Define path
    path = './data/' + organism +'/'

    if args.organism == "ecoli":
        organism_id = 3
    elif args.organism == "yeast":
        organism_id = 4

    test_size = 0.1
    validation_size = 0.1
    linkfile = path + "edgelist_biogrid.txt"
    datafile = path + "net" + str(organism_id) + "_expression_data.tsv"
    attrfile = path + "data_standard.txt"

    trainlinkfile = path + "edgelist_train.txt"
    testlinkfile = path + "edgelist_test_" + str(test_size) + ".txt"

    print("Creating test data")
    create_test_file(path, linkfile, trainlinkfile, testlinkfile, test_size=0.1)
    X_test = read_test_link(testlinkfile)

    print("Creating training and validation data")
    Data = data.LoadData(path, int(validation_size * 2018), validation_size, organism_id)
    
    print("Path: ", path)
    print("Total number of nodes: ", Data.id_N)
    print("Total number of attributes: ", Data.attr_M)
    print("Total training links: ", len(Data.links))
    print("Total epoch: ", args.epoch)
    print('Dimension of Structural Embedding (d):', args.id_dim)
    print('Dimension of Attribute Embedding (d):', args.attr_dim)
    print('Dimension of final representation (d):', args.attr_dim)

    # Learn embeddings 
    embeddings = run_GNE(path, Data, args)

    # Save embeddings 
    embeddings_file = open("./output/"+args.organism+"/embeddings_evaluation.pkl", 'wb')
    pickle.dump(embeddings, embeddings_file)
    embeddings_file.close()

    # compute similarity between embeddings 
    cosine_matrix = cosine_similarity(embeddings, embeddings)

    # Evaluate the prediction on test dataset
    roc, pr= evaluate_ROC_from_cosine_matrix(X_test, cosine_matrix)
    print("Accuracy using structure + attribute: {0:.9f}, {1:.9f} ".format( roc, pr))