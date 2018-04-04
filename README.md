# GNE: A deep learning framework for gene network inference by aggregating biological information

## Integrates gene interaction network with gene expression data to learn informative representations for gene network.

![](figures/gne.png)

## Architecture of GNE
![](figures/block_diagram.png)


## Datasets

| Dataset        | Source           | 
| ------------- |:-------------:|
| Interaction dataset  | [BioGRID](http://thebiogrid.org/) | 
| Gene expression data     | [DREAM5 Challenge](http://dreamchallenges.org/project/dream-5-network-inference-challenge/)    |  


## Implementation
Tensorflow implementation of Gene Network Embedding framework (GNE).

### Example to run the codes.
```
python GNE_runner.py --data_path path --id_dim 128 --attr_dim 128
```

## Contact
kk3671@rit.edu
