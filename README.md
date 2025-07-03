# Interpretable Predictions for Graph Classification using Variational Information Pursuit
**Andrei Vlad Dome**<br>
**`andreivlad.dome@gmail.com`**

This is the repository for my research project in my final semester at McMaster University, completed while visiting Ludwig Maximilian University of Munich, Chair for Mathematical Foundations of AI.

This code accompanies a research paper I wrote, which contains more detailed information about the project. It can be found in this repository as a PDF named `graph_vip_paper_dome_2025.pdf`

For any questions/inquiries related to the project, please don't hesitate to email me at `andreivlad.dome@gmail.com`!

Most of this work is based on the ICLR 2023 paper *Variational Information Pursuit for Interpretable Predictions* (Chattopadhyay et al. 2023). This paper can be found at this [link](https://arxiv.org/abs/2302.02876).


## Overview
<p align="center">
<img src="./assets/vip_graph_diagram_intro.png" alt="vip_graph_diagram_intro.png" width="500"/>
</p>

Graph Neural Networks (GNNs) have become ubiquitous for graph classification tasks. Although, their high accuracy comes with a sacrifice in interpretability, which is a pitfall in cases where transparency about a model’s decision-making process is important, such as when employed for scientific discovery purposes or in high-risk scenarios. Post-hoc explainability methods can be used to explain a black-box model’s internal reasoning, but may not always be reliable. As a result, models that are ”interpretable-by-design” have emerged. One such method is Variational Information Pursuit (V-IP), which is a neural network-based method that sequentially asks user-interpretable, task-relevant queries about data until a prediction can be made with some sufficient level of confidence, resulting in a sequence of queries and answers that provide full transparency about the model’s decision-making process *(See Figure Above)*. In this paper, we propose a framework for creating induced subgraph enumeration-based query sets for V-IP, in order to produce interpretable predictions for graph classification tasks. We demonstrate the efficacy of this framework by crafting a domain-specific query set for a graph classification task from chemistry, mutagen classification, and show that V-IP achieves test accuracies that beat those of black-box GNNs and another neural network-based interpretable-by-design model. Finally, we qualitatively show how V-IP’s explanations provide valuable insight into how certain functional groups (specific subgraphs) of a molecule play a role in it being classified as mutagenic or not, paving the way for future domain-centric research in Explainable AI for mutagen classification and other graph-based tasks.
<p align="center">
<img src="./assets/example_visualization.png" alt="example_visualization.png" width="400"/>
</p>

## Requirements
Please use the provided `requirements.txt` for installing all the required packages and their versions. `wandb` (Weights and Biases) is used to track training and testing performance. One may remove lines related to `wandb` and switch to other packages if they desire.

## Generating User-Defined Query Set
Before training the model, you must generate the queryset by running:

```
python3 generate_queryset.py \
  --dataset_root ./data/Mutagenicity \
  --save_dir  <SAVE_DIR>
```

where `<SAVE_DIR>` is the directory you wish to save the generated queryset, which will be saved as `rdkit_queryset.csv`.

The names of the functional groups in the queries are directly taken from the official documentation for the `rdkit.Chem.Fragments` module. Please refer to it at this [link](https://rdkit.org/docs/source/rdkit.Chem.Fragments.html) for more information about the functional groups used in the queryset.

## Training Mutagenicity
There are two stages of training: *Initial Random Sampling (IRS)* and *Subsequent Biased Sampling (SBS)*.

First, run IRS:

```
python3 train_mutagenicity.py \
  --epochs 100 \
  --batch_size 128 \
  --queryset_size 407 \
  --max_queries 407 \
  --max_queries_test 20 \
  --threshold 0.85 \
  --lr 0.0001 \
  --tau_start 1.0 \
  --tau_end 0.2 \
  --sampling random \
  --seed 0 \
  --name mutagenicity_random \
  --mode online \
  --save_dir <SAVE_DIR> \
  --data_dir <DATA_DIR> \
  --query_dir <QUERY_DIR>/rdkit_queryset.csv
```

Afterwards, run SBS:

```
python3 train_mutagenicity.py \
  --epochs 100 \
  --batch_size 128 \
  --queryset_size 407 \
  --max_queries 407 \
  --max_queries_test 20 \
  --threshold 0.85 \
  --lr 0.0001 \
  --tau_start 1.0 \
  --tau_end 0.2 \
  --sampling biased \
  --seed 0 \
  --name mutagenicity_biased \
  --mode online \
  --save_dir <SAVE_DIR> \
  --data_dir <DATA_DIR> \
  --query_dir <QUERY_DIR>/rdkit_queryset.csv
  --ckpt_path <CKPT_PATH>
```

where,
- `<SAVE_DIR>` is the folder to save model checkpoints during training
- `DATA_DIR` is the directory containing the raw Mutagenicity dataset
- `<QUERY_DIR>` is the folder containing the previously generated queryset as a CSV file
- `<CKPT_PATH>` is the path to the pre-trained model using IRS.

One can play around with the hyperparameters. Please refer to my research paper for hyperparameters used for certain experiments.

Code for performing a 10-fold Cross Validation, as is done in the research paper, is located in `train_mutagenicity_cv.py` and can be run using similar command line arguments as above, with additional arguments related to saving model checkpoints in a cross-validation loop.

## Checkpoints and Example Usage
Checkpoint to the model used to obtain the results in my research paper can be downloaded from this [HuggingFace link](https://huggingface.co/vladnotandrei/mutagen-classification-variational-information-pursuit/blob/main/example_model.ckpt) as a file called `example_model.ckpt`. A jupyter notebook named `example_usage.ipynb` with checkpoint loading instructions for the Mutagenicity dataset is located in the project's root folder.


<!-- | Dataset | OneDrive Link |
| :---: | :-------: |
| MNIST | [Link](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/kchan49_jh_edu/Ec1de2HcJ9dMuT9ScOhFsfcBeZ25A55rAo7lkdUMQpQoMg?e=PFvayh) |
| KMNIST | [Link](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/kchan49_jh_edu/EX5CH3HbXA5Eo1yIC7JMswAB5GaanEcRBDtd-kSjHOCEXw?e=UdHQYi) |
| Fashion MNIST | [Link](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/kchan49_jh_edu/EbocJetI_vpNmMZ0w33cQHIBGiH8_nxOT75YbfjP5ma47g?e=kAEevd) |
| Huffington News | [Link](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/kchan49_jh_edu/ETe8rbzfY0BKh5EmHe-mx-sBRRd_1BROEHEJU58O57my3g?e=7tXnnn) |
| Huffington News (cleaned)| [Link](git@github.com:ryanchankh/VariationalInformationPursuit.git)
| CUB-200 | [Link](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/kchan49_jh_edu/Eda0xGUGQ39Kl1d4LACN2agByKqByRMM0QZm6Rnibq4gBw?e=dXCcpw) |
| CUB-200 (concept) | [Link](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/kchan49_jh_edu/Ef3cdrFhegRFuqePkJJvOk0Bacw_lkh4iWl8rXECb7UrxA?e=V64Q5V) |  
| CIFAR10 | [Link](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/kchan49_jh_edu/ES_orEvtEc9Kjw4u1wgfiC8BvH7Y_6kaNVs-ZWvPqLcwjA?e=7a4Ylc) |
| SymCAT200 | [Link](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/kchan49_jh_edu/Edi9NVj6171DpfX4hgpJH3MB6xHxke2j7XRCunZKmb_CUw?e=SdTKxO) |
| SymCAT300 | [Link](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/kchan49_jh_edu/ERgvhBjxLj9GodXjFzGZTAAB4j0TP0EWd7EL1ZqL9eA_kQ?e=MAxemp) |
| SymCAT400 | [Link](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/kchan49_jh_edu/EbBJ8nDEk8dMmc6SXdh9rp0By5XG3Gf8Z0wLD8zHaJcXRw?e=gmJE68) | -->