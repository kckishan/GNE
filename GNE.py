'''
Tensorflow implementation of Gene Network Embedding framework (GNE)

@author: Kishan K C (kk3671@rit.edu)

'''

import pandas as pd
import random
import os
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
import evaluation

class GNE(BaseEstimator, TransformerMixin):
    def __init__(self, path, data, random_seed = 2018, parameters=None):
        # bind params to class
        # bind data to class
        self.path                   = path
        self.nodes                  = data.nodes
        self.node_neighbors_map     = data.node_neighbors_map
        self.node_N                 = data.id_N
        self.attr_M                 = data.attr_M
        self.X_train                = data.X
        
        # bind model parameters to class
        self.id_embedding_size      = parameters['id_embedding_size']
        self.attr_embedding_size    = parameters['attr_embedding_size']
        self.batch_size             = parameters['batch_size']
        self.alpha                  = parameters['alpha']
        self.n_neg_samples          = parameters['n_neg_samples']
        self.epoch                  = parameters['epoch']
        self.random_seed            = random_seed
        self.learning_rate          = parameters['learning_rate']
        self.representation_size    = parameters['representation_size']

        # init all variables in a tensorflow graph
        self._init_graph()
        print(parameters)

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default(), tf.device('/gpu:0'):
            # Set graph level random seed
            tf.set_random_seed(self.random_seed)

            # Define a placeholder for input data
            self.train_data_id          = tf.placeholder(tf.int32, shape=[None])  # batch_size * 1
            self.train_data_attr        = tf.placeholder(tf.float32, shape=[None, self.attr_M])  # batch_size * attr_M
            self.train_labels           = tf.placeholder(tf.int32, shape=[None, 1])  # batch_size * 1

            # Define placeholder for dropout 
            self.keep_prob              = tf.placeholder(tf.float32)

            # load initialzed variable.
            self.weights            = self._initialize_weights()

            # Model.
            # Look up embeddings for node_id. u = ENC(node_id)
            self.id_embed = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.weights['in_embeddings'], self.train_data_id) , 1) 

            # non linear transformation of expression data
            self.attr_embed = tf.nn.l2_normalize(tf.nn.elu(tf.matmul(self.train_data_attr, self.weights['attr_embeddings'])),1 ) 

            # Concatenation layer to concatenate structure and attribute
            self.embed_layer = tf.concat([self.id_embed, self.alpha * self.attr_embed], 1)

            # Non-linear transformation of concatenated representation
            self.representation_layer_dropout = tf.nn.dropout(self.embed_layer, self.keep_prob)
            self.representation_layer = tf.nn.tanh(tf.matmul(self.representation_layer_dropout, self.weights['hidden_weights']))

            # Compute the loss, using a sample of the negative labels each time.
            self.loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(self.weights['out_embeddings'], self.weights['biases'], self.train_labels, self.representation_layer, self.n_neg_samples, self.node_N))

            # Adam Optimizer.
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.loss)

            # init
            init        = tf.global_variables_initializer()
            self.sess   = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

            self.sess.run(init)

    def _initialize_weights(self):
        all_weights = dict()
        
        # Weights associated with structure embedding 
        all_weights['in_embeddings']    = tf.Variable(tf.random_uniform([self.node_N, self.id_embedding_size], -1.0, 1.0))  # id_N * id_dim

        # Weights associated with attribute embedding 
        all_weights['attr_embeddings']  = tf.Variable(tf.random_normal([self.attr_M, int(self.attr_embedding_size)]))  # attr_M * attr_dim
        
        # Weights and biases associated with neighborhood embedding 
        all_weights['out_embeddings']   = tf.Variable(tf.random_normal([self.node_N, self.representation_size]))
        all_weights['biases']           = tf.Variable(tf.zeros([self.node_N]))

        # Weights associated with hidden layer transformation
        all_weights['hidden_weights'] = tf.Variable(tf.random_normal([ self.id_embedding_size + self.attr_embedding_size, self.representation_size]))  # attr_M *eattr_dim
        return all_weights

    def partial_fit(self, X): 
        # Create a dictionary to feed to tensorflow graph
        feed_dict = {self.train_data_id: X['batch_data_id'], 
                     self.train_data_attr: X['batch_data_attr'],
                     self.train_labels: X['batch_data_label'], 
                     self.keep_prob : 0.5}
        
        # run the graph to compute loss                     
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss


    def train(self, validation_edges, validation_labels): # fit a dataset

        # Number of iterations executed
        total_iterations = 0

        # Best validation accuracy seen so far.
        best_validation_accuracy = 0.0

        # Iteration-number for last improvement to validation accuracy.
        last_improvement = 0

        # Stop optimization if no improvement found in this many iterations.
        require_improvement = 2

        print('Using structure and attribute embedding')
        for epoch in range( self.epoch ):
            # set the seed to randomize the permutation for each epoch
            random.seed(epoch)

            # random permutation of data
            perm = np.random.permutation(len(self.X_train['data_id_list']))
            self.X_train['data_id_list']    = self.X_train['data_id_list'][perm]
            self.X_train['data_attr_list']  = self.X_train['data_attr_list'][perm]
            self.X_train['data_label_list'] = self.X_train['data_label_list'][perm]

            # compute the number of batches using batch_size
            total_batch = int( len(self.X_train['data_id_list']) / self.batch_size)

            # Loop over all batches
            total_iterations += 1
            avg_cost = 0.
            for i in range(total_batch):
                # generate a batch data
                batch_xs = {}

                # set the seed to randomize the permutation for each batch within each epoch
                random.seed(epoch * i)
                start_index = np.random.randint(0, len(self.X_train['data_id_list']) - self.batch_size)
                batch_xs['batch_data_id']       = self.X_train['data_id_list'][start_index:(start_index + self.batch_size)]
                batch_xs['batch_data_attr']     = self.X_train['data_attr_list'][start_index:(start_index + self.batch_size)]
                batch_xs['batch_data_label']    = self.X_train['data_label_list'][start_index:(start_index + self.batch_size)]

                # Train the model with batch of data and compute the loss
                cost = self.partial_fit(batch_xs)
                avg_cost += cost / total_batch


            # Get embeddings from trained model
            Embeddings_out  = self.getEmbedding('out_embedding', self.nodes)
            Embeddings_in   = self.getEmbedding('embed_layer', self.nodes)
            Embeddings      = Embeddings_out + Embeddings_in

             # link prediction test
            ## If you want the embeddings for related gene to be closer to each other 
            ## adj_matrix_rec = -1*euclidean_distances(Embeddings, Embeddings)		
            ####################################################################################
            adj_matrix_rec = np.dot(Embeddings, Embeddings.T)
            roc, pr = evaluation.evaluate_ROC_from_matrix(validation_edges, validation_labels, adj_matrix_rec)

            attr_embeddings = self.getEmbedding('attribute', self.nodes)
            # If validation accuracy is an improvement over best-known.
            if roc > best_validation_accuracy:
                # Update the best-known validation accuracy.
                best_validation_accuracy = roc

                # Set the iteration for the last improvement to current.
                last_improvement = total_iterations

                # Save all variables of the TensorFlow graph to file.
                self.embedding_checkpoints(Embeddings, "save", "all")
                self.embedding_checkpoints(attr_embeddings, "save", "attribute")

                # A string to be printed below, shows improvement found.
                improved_str = '*'
            else:
                # An empty string to be printed below.
                # Shows that no improvement was found.
                improved_str = ''

            # Status-message for printing.
            msg = "Epoch: {0:>6}, Train-Batch Loss: {1:.9f}, Validation AUC: {2:.9f} {3}"
            print(msg.format(epoch + 1, avg_cost, roc, improved_str))

            # Early stopping: If no improvement found in the required number of iterations, stop training the model
            if total_iterations - last_improvement > require_improvement:
                print("No improvement found in a while, stopping optimization.")
                # Break out from the for-loop.
                break

        Embeddings = self.embedding_checkpoints(Embeddings, "restore", "all")
        attr_embeddings  = self.embedding_checkpoints(attr_embeddings, "restore", "attribute")
        return Embeddings, attr_embeddings

    def getEmbedding(self, type, nodes):
        # get the embedding
        if type == 'embed_layer':
            feed_dict = {self.train_data_id: nodes['node_id'], self.train_data_attr: nodes['node_attr'],
                         self.keep_prob: 1}
            Embedding = self.sess.run(self.representation_layer, feed_dict=feed_dict)
            return Embedding
        if type == 'out_embedding':
            Embedding = self.sess.run(self.weights['out_embeddings'])
            return Embedding
        if type == 'attribute':
            Embedding = self.sess.run(self.weights['attr_embeddings'])
            return Embedding
        if type == 'structure':
            Embedding = self.sess.run(self.weights['in_embeddings'])
            return Embedding

    def embedding_checkpoints(self, Embeddings, type, embedding_type="all"):
        file = self.path + "Embeddings_"+embedding_type+".txt"
        if type == "save":
            if os.path.isfile(file):
                os.remove(file)
            pd.DataFrame(Embeddings).to_csv(file, index=False, header=False)
        if type == 'restore':
            Embeddings = pd.read_csv(file, header=None)
            return np.array(Embeddings)
