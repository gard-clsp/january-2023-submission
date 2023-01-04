#!/bin/bash
set -e
ngpu=1
ncpu=1
CONDA_ROOT=/home/janto/usr/local/anaconda3
# CONDA_ROOT=/home/yshao/miniconda3
. $CONDA_ROOT/etc/profile.d/conda.sh

conda activate armory-k2
# conda activate snowfall_gard


if [ $ngpu -eq 0 ];then
    export CUDA_VISIBLE_DEVICES=""
else
    export CUDA_VISIBLE_DEVICES=$(free-gpu -n $ngpu)
fi
export LRU_CACHE_CAPACITY=1 #this will avoid crazy ram memory when using pytorch with cpu, it controls cache of MKLDNN
export OMP_NUM_THREADS=$ncpu
export MKL_NUM_THREADS=$ncpu

export ARMORY_GITHUB_TOKEN=$1

extra_args="--no-docker"

echo "run k2 snowfall undefended targeted pgd"
armory run --check $extra_args scenarios/JHUM_snowfall_undefended_targeted_pgd.json

echo "***************************************************************"
echo "run k2 snowfall undefended entailment"
armory run --check $extra_args scenarios/JHUM_snowfall_undefended_targeted_entailment.json

echo "***************************************************************"
echo "***************************************************************"
echo "run k2 snowfall denoiser targeted pgd"
armory run --check $extra_args scenarios/JHUM_snowfall_defended_denoiser_white_targeted_pgd.json

echo "***************************************************************"
echo "run k2 snowfall denoiser entailment"
armory run --check $extra_args scenarios/JHUM_snowfall_defended_denoiser_white_targeted_entailment.json

echo "***************************************************************"
echo "***************************************************************"
echo "***************************************************************"
echo "run k2 icefall undefended targeted pgd"
armory run --check $extra_args scenarios/JHUM_icefall_undefended_targeted_pgd.json

echo "***************************************************************"
echo "run k2 icefall undefended entailment"
armory run --check $extra_args scenarios/JHUM_icefall_undefended_targeted_entailment.json

echo "***************************************************************"
echo "***************************************************************"
echo "run k2 icefall denoiser targeted pgd"
armory run --check $extra_args scenarios/JHUM_icefall_defended_denoiser_white_targeted_pgd.json

echo "***************************************************************"
echo "run k2 icefall denoiser entailment"
armory run --check $extra_args scenarios/JHUM_icefall_defended_denoiser_white_targeted_entailment.json

echo "***************************************************************"
echo "***************************************************************"
echo "run k2 icefall denoiser targeted pgd"
armory run --check $extra_args scenarios/JHUM_icefall_defended_denoiser_chunking_white_targeted_pgd.json

echo "***************************************************************"
echo "run k2 icefall denoiser entailment"
armory run --check $extra_args scenarios/JHUM_icefall_defended_denoiser_chunking_white_targeted_entailment.json

