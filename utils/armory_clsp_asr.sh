#!/bin/bash

ncpu=4
ngpu=0
armory_opts=""
. utils/parse_options.sh || exit 1

cfg=$1

export ARM_ROOT=$(readlink -f `pwd -P`/../../..)
export TOOLS_ROOT=$ARM_ROOT/tools

# Add `libwarpctc.so` to LD_LIBRARY_PATH, required for deepspeech
#export LD_LIBRARY_PATH=/export/b17/janto/gard/armory-speech/egs-clsp/train-test-deepspeech-librispeech/v1/warp-ctc/build:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/export/b17/janto/gard/armory-speech/egs-clsp/train-test-deepspeech-librispeech/v1/warp-ctc-1.4/build:$LD_LIBRARY_PATH

#add hyperion to python path
HYP_ROOT=$TOOLS_ROOT/hyperion/hyperion
export PYTHONPATH=.:$HYP_ROOT:$PYTHONPATH

CONDA_ROOT=/home/janto/usr/local/anaconda3
#CONDA_ROOT=/home/yshao/miniconda3
. $CONDA_ROOT/etc/profile.d/conda.sh

#conda activate armory

# Activate  conda env for ASR eval
#conda activate armory-deepspeech
conda activate armory-deepspeech-t1.4
#conda activate armory-expresso
#conda activate snowfall

# gpu=$(awk '/use_gpu/ && /true/ { print true}' $cfg)
# if [ "$gpu" == "true" ];then
#     export CUDA_VISIBLE_DEVICES=$(free-gpu)
# else
#     export CUDA_VISIBLE_DEVICES=""
# fi
if [ $ngpu -eq 0 ];then
    export CUDA_VISIBLE_DEVICES=""
else
    export CUDA_VISIBLE_DEVICES=$(free-gpu -n $ngpu)
fi
    
export LRU_CACHE_CAPACITY=1 #this will avoid crazy ram memory when using pytorch with cpu, it controls cache of MKLDNN
export OMP_NUM_THREADS=$ncpu
export MKL_NUM_THREADS=$ncpu
echo $cfg
armory run --no-docker $armory_opts $cfg
