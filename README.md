# OSCAR: a Contention Window Optimization approach using Deep Reinforcement Learning
OSCAR: a Contention Window Optimization approach using Deep Reinforcement Learning to quickly learn the optimal policies under different network conditions

### Requisites
A working [ns3-gym](https://github.com/tkn-tub/ns3-gym) environment is required.
Moreover, a [wandb](https://wandb.ai/) account is required to show the results.

### Installation 
Clone this repository so that it lands inside ns3-gym/scratch/linear-mesh.
Edit ```run = wandb.init(entity="xraulz", project="contention_window", tags = wtags)``` in each training/test file and change ```entity=xraulz``` with ```entity=your_wandb_username```, where ```your_wand_username``` is the wandb account username.

### Start training 
Launch ```python OSCAR_train.py``` to train the OSCAR algorithm.

Launch ```python CCOD_train.py``` to train the CCOD algorithm.

Launch ```python standard_test_and_ccod_train.py``` to test the 802.11 algorithm and train the CCOD algorithm.

### Reference this paper
```
@INPROCEEDINGS{10279663,
  author={Grasso, Christian and Raftopoulos, Raoul and Schembra, Giovanni},
  booktitle={ICC 2023 - IEEE International Conference on Communications}, 
  title={OSCAR: A Contention Window Optimization Approach Using Deep Reinforcement Learning}, 
  year={2023},
  volume={},
  number={},
  pages={459-465},
  doi={10.1109/ICC45041.2023.10279663}}
```

