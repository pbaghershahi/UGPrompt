#!/bin/bash

pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121;
curr_version=$(python -c "import torch; print(torch.__version__)");
pytorch_version="${1:-$curr_version}";
pytorch_version="torch-${pytorch_version}.html";
pip install --no-index pyg_lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://pytorch-geometric.com/whl/$pytorch_version;
pip install torch-geometric;
pip install torcheval;
pip install torchmetrics;
pip install matplotlib;
pip install pandas;
pip install ipdb;
pip install gdown;
pip install notebook;
pip install pyyaml;
pip install -U "ray[data,train,tune,serve]";
pip uninstall fsspec -y;
pip install --force-reinstall -v "fsspec==2024.3.1"
printf "\033c";
