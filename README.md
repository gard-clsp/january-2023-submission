# JHU submission for GARD evaluation - Speech Commands Poisoning (Dirty Label) (January 2023)

This repo contains the code and scenarios for the evaluation.


## How to run in Docker

### First, install Docker app in your host machine, and run it

### Setting up API token to download the repo

Enter [GitHub Token settings](https://github.com/settings/tokens) and add a token with the access to private repos.
Set `ARMORY_GITHUB_TOKEN` env var with the token's value.

```bash
$  export ARMORY_GITHUB_TOKEN=...
```

### Installations

#### Installing armory
```bash
$ pip install armory-testbed
```

#### Installing other dependencies :
```bash
pip install tensorflow adversarial-robustness-toolbox Pillow 
pip install tensorflow_datasets==4.7.0 
pip install boto3 
pip install ffmpeg 
pip install librosa 
conda install -c nvidia cuda-cudart 
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install tidecv 
armory configure
```


#### Optional :
If you encounter troubles, you might need to install those librairies also :
```bash
apt install libgl1-mesa-glx
apt-get install gcc
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
$ bash build_docker.sh
```
<hr>

## Running scenarios

List of scenarios:

|  Scenario      | Description                                                           |
| -------------- | ----------------------------------------------------------------------|
| SUBMISSION0    | Baseline Pytorch model (undefended)                                   |
| SUBMISSION1    | DINO+KMeans+LDA Filtering defense                                     |

##  SUBMISSION0 : Baseline Pytorch model (undefended)
```bash run_jhu_poisoning_submission0.sh```
<hr>

## DEFENSE SUBMISSION1: DINO+KMeans+LDA Filtering defense

### [Step I] Dumping Training Data

```
$ bash run_dump_docker.sh
```

Optional: For custom dump path, set the `model.model_kwargs.dump_path` option in the scenario json file to path you desire.

Once the data dump has been performed, use the following command to move the dumped files into the docker of your choice:

```
$ docker cp dump_dir/* CONTAINER_NAME:/workspace/dump_dir
```

###  [Step II] Filter the poisoned examples
This step must be done inside the docker container.

#### Train the DINO model
*in the docker*, run the filtering of the poisoned examples:
```bash
$ cd /hyperion/egs/poison/dinossl.v1
$ bash RUN_ALL.sh scenario1 1 1 1
```
This will use the data previously dumped in <code>/workspace/dump_dir</code>, and augment with the musan noise in <code>/workspace/musan</code>,
to train a DINO network in a unsupervised way, produce DINO representations for the train dataset and filter them.

About the parameters:
- scenario1 is the base name for the scenario, but you can train an evaluate multiple ones in parallel in the same docker, just change the name of the data dump, and use a different scenario name.
- the first number is for the stage to start, they are 8 stages, if one crashes, you can start again at the wanted one using a different number. Dino training stage is the 6th, signature extraction is the 7th and computing the indexes to be removed is the 8th.
- the second number is for the number of GPUs you want to use, we only tested them up to 4GPU.
- the third number is optional, if you want to suppose 2 or more classes were simultaneously attacked, change this number.

The indices will be kept at a pickled lists in:
- <code>/workspace/scenario1_all.pkl</code> for the 1-step filtering
- <code>/workspace/scenario1_LDA_all.pkl</code> for the 2-step filtering with a LDA.
- <code>/workspace/scenario1_LDA_n1.pkl</code> for the 2-step filtering with a LDA, removing only the majority class (this is our current best).

###  [Step III] Run the evaluation
Finally, once we have the list of poisoned samples, we can run the evaluation.
```bash
docker cp JHUM_Armory_K2_Snowfall_Dockerfile:/workspace/scenario1_LDA_n1.pkl data_to_keep.pkl
bash run_jhu_poisoning_submission1.sh
```
