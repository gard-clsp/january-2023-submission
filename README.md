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

### Downloading noise dataset MUSAN
If you want to download musan for every instance, uncomment the last two lines of docker/JHUM_Armory_Hyperion_Dockerfile
If you already have it in a direction in MUSAN_PATH/musan, then build the docker and after that use :

```bash
$ docker cp MUSAN_PATH/musan CONTAINER_NAME:/workspace/
```
### Building the docker
To build the docker you are gonna use, launch :
```bash
$ ./build_docker.sh
```

## Running scenarios
### Dumping Training Data

$ ./run_dump.sh

For custom dump path, set the model.model_kwargs.dump_path option in the scenario json file.

Once the dump has been performed, use the following command to move the dumped files into the docker of your choice:

$ docker cp dump_dir/* CONTAINER_NAME:/workspace/poison_dump

### filter the poisoned examples
Then, in the docker, run the filtering of the poisoned examples:
```bash
$ cd /hyperion/egs/poison/dinossl.v1
$ ./RUN_ALL.sh /workspace/dump_dir scenario1 /workspace/musan retrain
```
This will use the data in */workspace/dump_dir*, augmented with the musan noise in */workspace/musan*,
to train unsupervisingly a DINO network, produce representations for the dataset and filter them.
The indices will be kept at a pickled list in */workspace/scenario1.pkl* and /workspace/scenario1_LDA.pkl*.

As the training takes 25 to 30 hours, if you wish to use a previously trained network, one can be found here:
```bash
https://drive.google.com/u/0/uc?id=1KMnknps7PsjuBZ3GPcDdiSHTWN_l8fFQ&export=download
```
You can use this to download it:
```bash
$ cd /hyperion/egs/poison/dinossl.v1
$ mkdir exp
$ mkdir exp/xvector_nnets
$ mkdir exp/xvector_nnets/fbank80_stmn_lresnet34_e256_do0_b48_amp.dinossl.v1
$ cd exp/xvector_nnets/fbank80_stmn_lresnet34_e256_do0_b48_amp.dinossl.v1
$ gdown https://drive.google.com/u/0/uc?id=1KMnknps7PsjuBZ3GPcDdiSHTWN_l8fFQ&export=download
$ cd ../../../
```

and then, run this instead, it will ignore the training of the network :

```bash
$ .RUN_ALL.sh /workspace/dump_dir scenario2 /workspace/musan no_train
```

### run the evaluation
Finally, once we have the list of poisoned samples, run the evaluation:
```bash
$ TODO
```
