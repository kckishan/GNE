from sklearn.preprocessing import MinMaxScaler

from convertdata import *
from numba import jit


class LoadData( object ):
    '''given the path of data, return the data format for GNE
    :param path
    return:
     X: a dictionary, ['data_id_list']-- len(links) of id for nodes in links ; 
                      ['data_attr_list']-- a list of attrs for corresponding nodes;
                     ['data_label_list']-- len(links) of neighbor for corresponding nodes

     nodes: a dictionary, ['node_id']--len(nodes) of id for nodes, one by one; ['node_attr']--a list of attrs for corresponding nodes
    '''

    # Three files are needed in the path
    def __init__(self, path, random_seed, test_size, organism_id, undirected = False):
        np.random.seed(random_seed)
        random.seed(random_seed)
        self.path = path
        self.undirected = undirected

        # Define files to work with
        self.linkfile = path + "edgelist_biogrid.txt"
        self.datafile = path + "net"+ str(organism_id) +"_expression_data.tsv"
        self.attrfile = path + "data_standard.txt"

        self.original_trainfile = "edgelist_train.txt"
        self.trainlinkfile = path + "edgelist_train_"+str(test_size)+".txt"
        
        self.vallinkfile = path + "edgelist_val_"+str(test_size)+".txt"

        # convert expression data to normalized,
        # Split train file to train and validation
        convertdata(self.path, self.datafile, self.original_trainfile, self.trainlinkfile, self.vallinkfile, test_size=test_size)
        
        self.node_map = {} 
        self.nodes = {}
        self.X = {}

        self.X_validation = [] # a list of 3-element lists (Gene_A, Gene_B, interaction(0/1)), read from vallinkfile
        self.node_neighbors_map = {} # [nodeid: neighbors_set] each node id maps to its neighbors set
        

        print("Constructing Nodes")
        self.construct_nodes()

        print("Constructing Neighborhood maps")
        self.construct_node_neighbors_map()

        # print("Aggregating feature from neighbors")
        self.create_node_features()

        print("Constructing train data")
        self.construct_X()

        print("Constructing validation links")
        self.read_validation_link()


    def readvocab(self):
        f = open(self.attrfile)
        self.vocab = {}
        line = f.readline()
        i = 0
        while line:
            items = line.strip().split(' ')
            for item in items[1:]:
                if item not in self.vocab:
                    self.vocab[ item ] = i
                    i = i + 1
            line = f.readline()
        f.close()
        self.attr_M = i
        print("attr_M:", self.attr_M)

    def readExp(self):
        f = open(self.attrfile)
        line = f.readline()
        items = line.strip().split(' ')
        self.attr_M = len(items[1:])
        print("attr_M:", self.attr_M)

    def construct_nodes(self):
        '''construct the dictionary '''
        self.readExp()
        f = open(self.attrfile)
        i = 0
        self.nodes['node_id'] = []
        line = f.readline()
        while line:
            line = line.strip().split(' ')
            self.node_map[int(line[0])] = i # map the node
            self.nodes['node_id'].append(i) # only put id in nodes, not the original name
            i = i + 1
            line = f.readline()
        f.close()
        self.id_N = i
        print("id_N:", self.id_N)

    def read_link(self): # read link file to a list of links
        f = open(self.trainlinkfile)
        self.links = []
        line = f.readline()
        while line:
            line = line.strip().split(' ')
            link = [int(line[0]), int(line[1])]
            self.links.append(link)
            if self.undirected:
                link = [int(line[1]), int(line[0])]
                self.links.append(link)
            line = f.readline()
        f.close()


    @jit
    def construct_X(self):
        self.X['data_id_list'] = np.ndarray(shape=(len(self.links)), dtype=np.int32)
        self.X['data_attr_list'] = np.ndarray(shape=(len(self.links),  self.attr_M), dtype=np.float32)
        self.X['data_label_list'] = np.ndarray(shape=(len(self.links), 1), dtype=np.int32)

        for i in range(len(self.links)):
            self.X['data_id_list'][i] = int(self.node_map[self.links[i][0]])
            self.X['data_attr_list'][i] = self.nodes['node_attr'][self.links[i][0]]  # dimension need to change to  self.attr_dim
            self.X['data_label_list'][i, 0] = int(self.node_map[self.links[i][1]])  # one neighbor of the node

    def construct_node_neighbors_map(self):
        self.read_link()
        for link in self.links:
            if self.node_map[ link[0] ] not in self.node_neighbors_map:
                self.node_neighbors_map[self.node_map[ link[0] ]] = set( [self.node_map[ link[1] ]] )
            else:
                self.node_neighbors_map[self.node_map[ link[0] ]].add(self.node_map[ link[1] ])

    def read_test_link(self):
        f = open(self.testlinkfile)
        line = f.readline()
        while line:
            line = line.strip().split(' ')
            self.X_test.append([self.node_map[ int(line[0]) ] ,  self.node_map[ int(line[1]) ] ,  int(line[2])  ] )
            line = f.readline()
        f.close()
        print("test link number:", len(self.X_test))

    def read_validation_link(self):
        f = open(self.vallinkfile)
        line = f.readline()
        while line:
            line = line.strip().split(' ')
            self.X_validation.append([self.node_map[int(line[0])], self.node_map[int(line[1])], int(line[2])])
            line = f.readline()
        f.close()
        print("Validation link number:", len(self.X_validation))

    # @jit
    def create_node_features(self):
        '''
        Creating node features
        Also, add neighborhood features but not used in the model
        '''
        self.nodes['node_attr'] = np.zeros([len(self.nodes['node_id']), self.attr_M])
        self.nodes['node_neigh_attr'] = np.zeros([len(self.nodes['node_id']), self.attr_M])
        no_neighbor = np.zeros([self.attr_M, ])
        neighbor_vecs = np.zeros([1, self.attr_M])

        exp = np.array(pd.read_csv(self.attrfile, sep=" ", header=None))
        scaler = MinMaxScaler()
        exp = scaler.fit_transform(exp)
        for node_id in self.nodes['node_id']:
            vs = exp[node_id,1:]

            if node_id not in self.node_neighbors_map:
                neighbor_vecs = no_neighbor
            else:
                neighbor_ids = list(self.node_neighbors_map[node_id])

                # features from sampled neighborhood
                neighbor_attr = exp[neighbor_ids,1:]
                #mean aggregator
                neighbor_vecs = np.array(np.mean(neighbor_attr, axis=0))

                ## max aggregator
                # neighbor_vecs = np.max(neighbor_attr, axis=0)
            self.nodes['node_attr'][node_id] = vs
            self.nodes['node_neigh_attr'][node_id] = neighbor_vecs
