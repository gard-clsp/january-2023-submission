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
| SUBMISSION2.1  | Sliding Joint Energy-based Model defense with model training          |
| SUBMISSION2.2  | Sliding Joint Energy-based Model defense with loading model           |

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

There are two options for this step. [OPTION A] will re-train the DINO model from scratch, while [OPTION B] will use the model we already trained for the default scenario (10% of source class 11 poisoned with <code>clapping.wav</code> trigger to target class 2)

#### [OPTION A] Retrain the DINO model
*in the docker*, run the filtering of the poisoned examples:
```bash
$ cd /hyperion/egs/poison/dinossl.v1
$ bash RUN_ALL.sh retrain
```
This will use the data previously dumped in <code>/workspace/dump_dir</code>, and augment with the musan noise in <code>/workspace/musan</code>,
to train a DINO network in a unsupervised way, produce DINO representations for the train dataset and filter them.
The indices will be kept at a pickled list in <code>/workspace/scenario1.pkl</code> and <code>/workspace/scenario1_LDA.pkl</code>.

#### [OPTION B] Load the DINO model (already trained) to save time
As the training takes 25 to 30 hours, if you wish to use a previously trained network, one can be found here:
```bash
https://drive.google.com/u/0/uc?id=1KMnknps7PsjuBZ3GPcDdiSHTWN_l8fFQ&export=download
```
You can use this to download it:
```bash
$ cd /hyperion/egs/poison/dinossl.v1/exp/xvector_nnets/fbank80_stmn_lresnet34_e256_do0_b48_amp.dinossl.v1
$ gdown https://drive.google.com/u/0/uc?id=1KMnknps7PsjuBZ3GPcDdiSHTWN_l8fFQ&export=download
```

and then, run this instead, it will ignore the training of the network :

```bash
$ cd /hyperion/egs/poison/dinossl.v1
$ bash RUN_ALL.sh no_train
```

###  [Step III] Run the evaluation
Finally, once we have the list of poisoned samples, we can run the evaluation.
```bash
docker cp JHUM_Armory_K2_Snowfall_Dockerfile:/workspace/scenario1_LDA.pkl data_to_keep.pkl
bash run_jhu_poisoning_submission1.sh
```

<hr>

## DEFENSE SUBMISSION2: Sliding Joint Energy-based Model defense
### [OPTION A] Training the model

```bash run_jhu_poisoning_submission2.1.sh```
<br>
Note: You might need to change ```variant_path``` [here](https://github.com/gard-clsp/january-2023-submission/blob/main/scenario_configs_eval6_v1/jhu_defense_slidingJEM/poisoning_v0_audio_p10_jem_pytorch_v1.json#L42) if the current working directory has changed.

### [OPTION B] Loading already trained model

Evaluates already trained model (10% of source class 11 poisoned with clapping trigger, target class 2) <br>
```bash run_jhu_poisoning_submission2.2.sh```
<br>
Note: You might need to change the ```model_path``` [here](https://github.com/gard-clsp/january-2023-submission/blob/main/scenario_configs_eval6_v1/jhu_defense_slidingJEM/poisoning_v0_audio_p10_jem_pytorch_load_model.json#L42) if the current working directory has changed.


## DEFENSE SUBMISSION3: Combination of SUBMISSION1 and SUBMISSION2

This defense applies filtering used in SUBMISSION1 and trains the model from SUBMISSION2. It expects file data_to_keep.pkl. If you didn’t run SUBMISSON1, follow steps until you obtain data_to_keep.pkl.
### [OPTION A] Training the model

```bash run_jhu_poisoning_submission3.1.sh```
<br>
Note: You might need to change ```variant_path``` [here](https://github.com/gard-clsp/january-2023-submission/blob/main/scenario_configs_eval6_v1/jhu_defense_jem_filtering/poisoning_v0_audio_p10_jem_filter.json#L51) if the current working directory has changed.

### [OPTION B] Loading already trained model
Evaluates already trained model (10% of source class 11 poisoned with clapping trigger, target class 2) <br>
```bash run_jhu_poisoning_submission3.2.sh```
<br>
Note: You might need to change the ```model_path``` [here](https://github.com/gard-clsp/january-2023-submission/blob/main/scenario_configs_eval6_v1/jhu_defense_jem_filtering/poisoning_v0_audio_p10_jem_filter_load_model.json#L51) if the current working directory has changed.
