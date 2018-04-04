from sklearn.linear_model import LogisticRegression

import LoadData as data
from evaluation import *
from GNE import GNE
from convertdata import *

################################# Define parameters to train GNE model #######################################
parameters = {}
parameters['id_embedding_size'] = 128
parameters['attr_embedding_size'] = 128
parameters['alpha'] = 1
parameters['n_neg_samples'] = 10
parameters['epoch'] = 20
parameters['representation_size'] = 128
parameters['batch_size'] = 256
parameters['learning_rate'] = 0.005

print(parameters)

################################################################################################################


#################################### Define dataset and files ##################################################
# Define dataset to run the model: yeast or ecoli
organism = 'yeast'
# Define path
path = './data/' + organism +'/'

geneids = pd.read_csv(path + "gene_ids.tsv", sep=" ")
num_genes = geneids.shape[0]

# Define the input to GNE model
link_file = path + "edgelist_biogrid.txt"
feature_file = path + 'expression_data.tsv'

################################################################################################################


################################# Load network and split to train and test######################################

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
print("Total genes:", adj.shape[0])
print("Training interactions (positive):", len(train_edges))
print("Training interactions (negative):", len(train_edges_false))
print("Validation interactions (positive):", len(val_edges))
print("Validation interactions (negative):", len(val_edges_false))
print("Test interactions (positive):", len(test_edges))
print("Test interactions (negative):", len(test_edges_false))

################################################################################################################


###################### Combine positive and negative interactions for valdiation and test ######################
# Create validation edges and labels
validation_edges = np.concatenate([val_edges, val_edges_false])
val_edge_labels = np.concatenate([np.ones(len(val_edges)), np.zeros(len(val_edges_false))])

# Create test edges and labels
test_edges_data = np.concatenate([test_edges, test_edges_false])
test_edge_labels = np.concatenate([np.ones(len(test_edges)), np.zeros(len(test_edges_false))])

################################################################################################################


################## load interaction and expression data to fit GNE model and learn embeddings ##################
# load dataset to fit GNE model
Data = data.LoadData(path, train_links=train_edges, features_file=feature_file)

# Define GNE model with data and parameters
model = GNE(path, Data, 2018, parameters)

# learn embeddings
embeddings, attr_embeddings = model.train(validation_edges, val_edge_labels)

################################################################################################################


################## Create feature matrix and true labels for training and randomize the rows  ##################
# Train-set edge embeddings
pos_train_edge_embs = get_edge_embeddings(embeddings, train_edges)
neg_train_edge_embs = get_edge_embeddings(embeddings, train_edges_false)
train_edge_embs = np.concatenate([pos_train_edge_embs, neg_train_edge_embs])

# Create train-set edge labels: 1 = real edge, 0 = false edge
train_edge_labels = np.concatenate([np.ones(len(train_edges)), np.zeros(len(train_edges_false))])

# Randomize train edges and labels
index = np.random.permutation([i for i in range(len(train_edge_labels))])
train_data = train_edge_embs[index, :]
train_labels = train_edge_labels[index]

################################################################################################################


################## Train the logistic regression on training data and predict on test dataset ##################
# Train logistic regression on train-set edge embeddings
edge_classifier = LogisticRegression(random_state=0)
edge_classifier.fit(train_data, train_labels)

# Test-set edge embeddings, labels
pos_test_edge_embs = get_edge_embeddings(embeddings, test_edges)
neg_test_edge_embs = get_edge_embeddings(embeddings, test_edges_false)
test_edge_embs = np.concatenate([pos_test_edge_embs, neg_test_edge_embs])

# Randomize test edges and labels
index = np.random.permutation([i for i in range(len(test_edge_labels))])
test_data = test_edge_embs[index, :]
test_labels = test_edge_labels[index]

# Predict the probabilty for test edges by trained classifier
test_preds = edge_classifier.predict_proba(test_data)[:, 1]
test_roc = roc_auc_score(test_labels, test_preds)
test_ap = average_precision_score(test_labels, test_preds)

msg = "Alpha: {0:>6}, GNE Test ROC Score: {1:.9f}, GNE Test AP score: {2:.9f}"
print(msg.format(parameters['alpha'], test_roc, test_ap))

################################################################################################################


########################################## Save the embedding to a file ########################################

embeddings_file = open(path + "embeddings_trainsize_alpha_"+str(parameters['alpha'])+".pkl", 'wb')
pickle.dump(embeddings, embeddings_file)
embeddings_file.close()

################################################################################################################
