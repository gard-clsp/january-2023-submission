#!/bin/bash
set -e
ngpu=0
ncpu=1
CONDA_ROOT=/Users/thomas/miniconda3
. $CONDA_ROOT/etc/profile.d/conda.sh
conda activate dino

if [ $ngpu -eq 0 ];then
    export CUDA_VISIBLE_DEVICES=""
    extra_args="--no-gpu"
else
    export CUDA_VISIBLE_DEVICES=$(free-gpu -n $ngpu)
fi
export LRU_CACHE_CAPACITY=1 #this will avoid crazy ram memory when using pytorch with cpu, it controls cache of MKLDNN
export OMP_NUM_THREADS=$ncpu
export MKL_NUM_THREADS=$ncpu

export ARMORY_GITHUB_TOKEN=$1


echo "run data dump"
armory run --check $extra_args scenario_configs_eval6_v1/jhu_dump/poisoning_v0_audio_p10_trigger2_undefended_pytorch_dump.json

