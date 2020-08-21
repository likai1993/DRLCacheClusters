# Deep Reinforcement Learning-Based Cache Replacement Policy

## Introduction

We adopt a learning-based method to cache replacement strategy, aiming to improve the miss rate of existing traditional cache replacement policies. The main idea of modeling is to regard the strategy as a MDP so that we can employ DRL to learn how to make decision. We refer to Zhong et al. and design a similar MDP model. The learning backbone, however, is value-based DQN. Our main effort is to use short-term reward to optimize the long term miss rate, and further reducing the model size by clustering the cache pages and using the cluster features, which makes our work pratical and compatible with real-world systems.

## Code

### Dependency
To run the code for experiment, you should have the following dependencies package installed at least. We have tested and run through our code with Python 3.6 on MacOSX and Ubuntu 16.04.
```
numpy
scipy
pandas
tensorflow1.0
```

### Implementation
The modules are categorized into directory `agents` and `cache`. The `agents` folder contains our implementation of DRL agent and reflex agents, while the `cache` folder contains a cache simulator and its affiliated data loader.

* `CacheAgent.py` contains a series of base classes of cache replacement agents.

* `ReflexAgent.py` contains our implementation of cache agents of hand-crafted replacement policy, e.g. LRU, Random, etc.

* `DQNAgent.py` contains class `DQNAgent`, a cache agent with DRL-based replacement strategy. `DQNAgent` is based on Deep Q-Network and we employ `tensorflow` to build the MLPs.

* `Cache.py` contains a simulated cache system, acting as the environment for every agent. It not only maintains cache states, but also receives actions from agents and gives feedbacks. Hence, it accepts multiple set of parameters not only to setup the cache system itself, but also to specify the observation features and reward functions.

* `DataLoader.py` contains two subclasses `DataLoaderPintos` and `DataLoaderZipf`.

    * `DataLoaderPintos` can load data from our collected or synthetic dataset saved in `.csv` format. Refer to our dataset for details
    * `DataLoaderZipf` can generate access records by mimicking disk activities using Zipf distribution.
    * `DataLoaderMix` can generate mixed access requests by peroidically fetching requests (e.g., 10 requests of each) from two trace files.

### Dataset 
* `training_data` contains traces which can be used to train the model. It consists of two patterns of trace, one is zipf-related (e.g., 80% requests fall into 20% content), the other features as pure-sequential scan (given a disk address range, randomly pick a starting point, and then randomly pick the length for a scan and so on repeats). The difference of filename between the two patterns is that the second one contains `_seq` in the filename.
 
* `testing_data` contains traces which can be used to test our trained model.

### Experiments

* To train/test a pure zipf trace. 
```bash
    ./run_training_pure.py training_data/zipf_15k_40k.data
    ./run_inference_pure.py testing_data/zipf_15k_40k.data
```

* To train/test a pure sequential-scan trace. 
```bash
    ./run_training_pure.py training_data/zipf_15k_40k_seq.data
    ./run_inference_pure.py testing_data/zipf_15k_40k_seq.data
```

* To train/test the mixed of above traces. 
```bash
    ./run_training_mixed.py training_data/zipf_15k_40k.data training_data/zipf_15k_40k_seq.data
    ./run_inference_mixed.py testing_data/zipf_15k_40k.data testing_data/zipf_15k_40k_seq.data 
```

* To run classic algorithms over the traces. 
```bash
    ./run_classic_algos.py testing_data/zipf_15k_40k.data 
    ./run_classic_algos.py testing_data/zipf_15k_40k_seq.data 
    ./run_classic_algos.py testing_data/zipf_15k_40k.data testing_data/zipf_15k_40k_seq.data
```
Note you can modify the cache size in the above python scripts.

## Acknowledgment
Partial of the codebase is based on the open source code from [Peihao Wang](https://github.com/peihaowang/DRLCache).
