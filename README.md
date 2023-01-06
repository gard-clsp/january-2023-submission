# ASR submission for GARD evaluation (January 2023)

This repo contains the code and scenarios for the evaluation.

## How to run in Docker

### First, install Docker app in your host machine, and run it

### Setting up API token to download the repo

Enter [GitHub Token settins](https://github.com/settings/tokens) and add a token with the access to private repos.
Set `ARMORY_GITHUB_TOKEN` env var with the token's value.

```bash
$  export ARMORY_GITHUB_TOKEN=...
```

### Installing armory

```bash
$ pip install armory-testbed==0.16.0
```


## Running scenarios

### Dumping data
First we must create the poisoned dataset and dump it in the docker:
```bash
$ .run_dump_docker.sh
```
### filter the poisoned examples
Then, in the docker, run the filtering of the poisoned examples:
```bash
$ cd /hyperion/egs/poison/dinossl.v1
$ .RUN_ALL.sh /workspace/poison_dump scenario1 /workspace/musan retrain
```
This will use the data in */workspace/poison_dump*, augmented with the musan noise in */workspace/musan*,
to train unsupervisingly a DINO network, produce representations for the dataset and filter them.
The indices will be kept at a pickled list in */workspace/scenario1.pkl* and /workspace/scenario1_LDA.pkl*.

As the training takes 25 to 30 hours, if you wish to use a previously trained network, one can be found here: 
https://drive.google.com/file/d/1KMnknps7PsjuBZ3GPcDdiSHTWN_l8fFQ/view?usp=sharing 
download it and put it in here:
```bash
/hyperion/egs/poison/dinossl.v1/exp/xvector_nnets/fbank80_stmn_lresnet34_e256_do0_b48_amp.dinossl.v1/
```
and run this instead, it will ignore the training of the network :
```bash
$ cd /hyperion/egs/poison/dinossl.v1
$ .RUN_ALL.sh /workspace/poison_dump scenario2 /workspace/musan
```

### run the evaluation
Finally, once we have the list of poisoned samples, run the evaluation:
```bash
$ TODO
```


