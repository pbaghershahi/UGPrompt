
# UGPROMPT: Unsupervised Prompting for Graph Neural Networks

This is a PyTorch implementation of the paper.

## Requirements

To install the requirements, please use the "installation.sh" file. Run the following:
```
chmod +x installation.sh
./installation.sh
```

## Train Base GNNS and Prompt Models

To train a base GNN model and train a prompting method run the main.py file. For example to run train a GCN model on the PubMed dataset, fix it, and train UGPrompt for prompting run the following command:
```
python main.py --pretrain --config-from-file ./config/pubmed.yaml --total-iters 10 --num-runs 5
```

This will run 50 experiments with the setting and report the average performance; that is 10 different initialization of the dataset (```--total-iters 10```) and 5 rounds of initializing the prompting function parameters (```--num-runs 5```).

The settings for all the datasets are provided in the ```config``` directory with the name of the datasets. To change the datasets change the path for ```--config-from-file```.

## Train Prompt Models for Trained Base GNNs
To train prompt models for trained base GNNs use the arguments ```seed``` and ```pretrained-path```. The following example is the original experiment reported in the paper with 10 different seeds for initializing PubMed datasets and using the correspoding base GCN models:
```
python main.py --config-from-file ./config/pubmed.yaml --total-iters 10 --num-runs 5 \\
--seed 3170 2470 2044 3616 4152 4700 4941 2529 4934 4375 \\
--pretrained-path ./pretrained/GCN_Pretrained_2024-07-19-22-45-14.pth ./pretrained/GCN_Pretrained_2024-07-19-22-55-45.pth ./pretrained/GCN_Pretrained_2024-07-19-23-07-06.pth ./pretrained/GCN_Pretrained_2024-07-19-23-18-19.pth ./pretrained/GCN_Pretrained_2024-07-19-23-29-41.pth ./pretrained/GCN_Pretrained_2024-07-19-23-40-55.pth ./pretrained/GCN_Pretrained_2024-07-19-23-52-08.pth ./pretrained/GCN_Pretrained_2024-07-20-00-03-23.pth ./pretrained/GCN_Pretrained_2024-07-20-00-14-46.pth ./pretrained/GCN_Pretrained_2024-07-20-00-26-14.pth
```

## Supported Datasets
Dataset | path |
:--- | :---: |
ENZYMES | ```./config/enzymes.yaml``` |
PROTEINS | ```./config/proteins.yaml``` |
DHFR | ```./config/dhfr.yaml``` |
Cora | ```./config/cora.yaml``` |
CiteSeer | ```./config/citeseer.yaml``` |
PubMed | ```./config/pubmed.yaml``` |

## Supported Prompting Methods
Methods | method name |
:--- | :---: |
UGPrompt | ```fix_match```/```flex_match``` |
GPF-Plus | ```gpf_plus``` |
All-In-One | ```all_in_one``` |

To change the prompting function use the ```--prompt-method``` argument.

