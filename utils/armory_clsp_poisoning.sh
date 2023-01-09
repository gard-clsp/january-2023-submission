#!/bin/bash

ncpu=4
ngpu=0
armory_opts=""
. utils/parse_options.sh || exit 1

cfg=$1

### Sonal added reset paths

export PATH=":/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games"
export PATH=$PATH:/usr/local/cuda/bin/
unset PYTHONPATH
unset CONDA_ROOT
unset LD_LIBRARY_PATH

##

export ARM_ROOT=$(readlink -f `pwd -P`/../../..)
export TOOLS_ROOT=$ARM_ROOT/tools

# Add `libwarpctc.so` to LD_LIBRARY_PATH, required for deepspeech
#export LD_LIBRARY_PATH=/export/b17/janto/gard/armory-speech/egs-clsp/train-test-deepspeech-librispeech/v1/warp-ctc-1.4/build:$LD_LIBRARY_PATH

# Required by armory Tensorflow
#export LD_LIBRARY_PATH=/home/smielke/cuda-cudnn/lib64/libcudnn.so
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/smielke/cuda-cudnn/lib64
export LD_LIBRARY_PATH=/export/c11/sjoshi/miniconda3/envs/armory_Oct22_poisoning/lib:$LD_LIBRARY_PATH

# Add hyperion to python path
#HYP_ROOT=$TOOLS_ROOT/hyperion/hyperion
#export PYTHONPATH=.:$HYP_ROOT:$PYTHONPATH

# Set conda root
CONDA_ROOT=/Users/thomas/miniconda3 #/root/anaconda3
. $CONDA_ROOT/etc/profile.d/conda.sh

# Activate  conda env for poisoning
#conda activate armory_Oct22_poisoning
conda activate dino
#conda activate armory_Oct22_poisoning_v2

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

echo "Testing if GPU is seen in Tensorflow"
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
echo "====================================================="
armory run --no-docker $armory_opts $cfg
