# BNU-SVIBD
This project is the official implementation of our paper Classroom Interaction Behaviour Recognition Dataset, authored by Bo Sun, Minglin Hong, Jun He, Yinghui Zhang, Chuang Wang, Jing Sun.

## Dependencies

- Python `3.x`
- PyTorch `1.7.1`
- numpy, pickle, scikit-image
- [RoIAlign for Pytorch](https://github.com/longcw/RoIAlign.pytorch)
- Datasets:

## Get Started

1. Stage1: Fine-tune the model on single frame without using GCN.

    ```shell
    python train_interactive_stage1.py
    ```

2. Stage2: Fix weights of the feature extraction part of network, and train the network with GCN.

    ```shell
    python train_interactive_stage2.py
    ```

## Acknowledgement
We are very grateful to the authors of 
[Group-Activity-Recognition](https://github.com/wjchaoGit/Group-Activity-Recognition) for open-sourcing their code from which this repository is heavily sourced. If your find this research useful, please consider citing their paper as well.
```
@inproceedings{CVPR2019_ARG,
  title = {Learning Actor Relation Graphs for Group Activity Recognition},
  author = {Jianchao Wu and Limin Wang and Li Wang and Jie Guo and Gangshan Wu},
  booktitle = {CVPR},
  year = {2019},
}
```
