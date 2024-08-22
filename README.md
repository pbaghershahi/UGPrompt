
# UGPROMPT: Unsupervised Prompting for Graph Neural Networks

This is a PyTorch implementation of the paper.

## Requirements

To install the requirements, please use the "installation.sh" file. Run the following:
```
chmod +x installation.sh
./installation.sh
```

## Usage

To train a base GNN model and train a prompting method run the main.py file. For example to run train a GCN model on the PubMed dataset, fix it, and train UGPrompt for prompting run the following command:
```
python main.py --pretrain --config-from-file ./config/pubmed.yaml --total-iters 10 --num-runs 5
```

This will run 50 experiments with the setting and report the average performance; that is 10 different initialization of the dataset (```--total-iters 10```) and 5 rounds of initializing the prompting function parameters (```--num-runs 5```).

The settings for all the datasets are provided in the ```config``` directory with the name of the datasets. To change the datasets change the path for ```--config-from-file```.

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
All-In-One | ```all_in_one_original``` |

To change the prompting function use the ```--prompt-method``` argument.
