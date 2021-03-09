# mvgrl
This DGL example implements the model proposed in the paper [Contrastive Multi-View Representation Learning on Graphs](hhttps://arxiv.org/abs/2006.05582).

Paper link: https://arxiv.org/abs/2006.05582
Author's code: https://github.com/kavehhassani/mvgrl

## Dependencies

- Python 3.7
- PyTorch 1.7.1
- dgl 0.6.0

## Datasets

##### Unsupervised Graph Classification Datasets:

 'MUTAG', 'PTC', 'IMDBBINARY', 'IMDBMULTI', 'REDDITBINARY', 'REDDITMULTI5K' of dgl.data.GINDataset.

| Dataset         | MUTAG | PTC   | RDT-B  | RDT-M5K | IMDB-B | IMDB-M |
| --------------- | ----- | ----- | ------ | ------- | ------ | ------ |
| # Graphs        | 188   | 344   | 2000   | 4999    | 1000   | 1500   |
| # Classes       | 2     | 2     | 2      | 5       | 2      | 3      |
| Avg. Graph Size | 17.93 | 14.29 | 429.63 | 508.52  | 19.77  | 13.00  |

##### Unsupervised Node Classification Datasets:

'Cora', 'Citeseer' and 'Pubmed'

| Dataset  | # Nodes | # Edges | # Classes |
| -------- | ------- | ------- | --------- |
| Cora     | 2,708   | 10,556  | 7         |
| Citeseer | 3,327   | 9,228   | 6         |
| Pubmed   | 19,717  | 88,651  | 3         |


## Arguments

##### 	Graph Classification:



```
--dataname         str     The graph dataset name.                Default is 'MUTAG'.
--gpu              int     GPU index.                             Default is -1, using cpu.
--epochs           int     Number of training epochs.             Default is 20.
--batch_size       int     Size of a training batch.              Default is 128.
--patience         int     Early stopping steps.                  Default is 20.
--lr1              float   Learning rate of main model.           Default is 0.001.
--lr2              float   Learning rate of linear classifer.     Default is 0.01.
--wd1              float   Weight decay of main model.            Default is 0.0.
--wd2							 float   Weight decay of linear classifier.     Default is 0.0.
--hid_dim          float   Embedding dimension.                   Default is 512.
```

##### 	Node Classification:

```
--dataname         str     The graph dataset name.                Default is 'cora'.
--gpu              int     GPU index.                             Default is -1, using cpu.
--epochs           int     Number of training epochs.             Default is 20.
--batch_size       int     Size of a training batch.              Default is 128.
--patience         int     Early stopping steps.                  Default is 20.
--lr1              float   Learning rate of main model.           Default is 0.001.
--lr2              float   Learning rate of linear classifer.     Default is 0.01.
--wd1              float   Weight decay of main model.            Default is 0.0.
--wd2							 float   Weight decay of linear classifier.     Default is 0.0.
--hid_dim          float   Embedding dimension.                   Default is 512.
```



## How to run examples

Training and testing unsupervised model on MUTAG.(We recommend using cpu)
```bash
# MUTAG:
python unsupervised.py --dataname MUTAG --n_layers 4 --hid_dim 32
```
Replace 'MUTAG' with dataname in [MUTAG', 'PTC', 'IMDBBINARY', 'IMDBMULTI', 'REDDITBINARY', 'REDDITMULTI5K'] if you'd like to try other datasets.

Training and testing semi-supervised model on QM9 for graph property 'mu' with gpu.

```bash
# QM9:
python semisupervised.py --gpu 0 --target mu
```

Replace 'mu' with other target names above

## 	Performance

We use the same  hyper-parameter settings as stated in the original paper.

##### Graph Classification:

|      Dataset      | MUTAG |  PTC  | REDDIT-B | REDDIT-M | IMDB-B | IMDB-M |
| :---------------: | :---: | :---: | :------: | -------- | ------ | ------ |
| Accuracy Reported | 89.01 | 61.65 |  82.50   | 53.46    | 73.03  | 49.69  |
|        DGL        | 89.88 | 63.54 |  88.50   | 56.27    | 72.70  | 50.13  |

##### Node classification:

|      Dataset      |  Cora  | Citeseer | Pubmed |
| :---------------: | :----: | :------: | :----: |
| Accuracy Reported | 0.3169 |  0.5444  | 0.0060 |
|       DGL-1       | 0.2411 |  0.5192  | 0.1560 |
|       DGL-2       | 0.2355 |  0.5483  | 0.1581 |

