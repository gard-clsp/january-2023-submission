#!/bin/bash
# Copyright
#                2020   Johns Hopkins University (Author: Jesus Villalba, Sonal Joshi)
# Apache 2.0.
#
. ./cmd.sh
set -e

ncpu=1
ngpu=0
armory_opts=""

. utils/parse_options.sh || exit 1

if [ $ngpu -eq 0 ]; then
    cmd="$cpu_cmd --num-threads $ncpu"
else
    cmd="$cuda_cmd -l gpu=$ngpu --num-threads $ncpu"
fi

echo "Run eval for DINO Clustering Filtering defense"
echo "Note: This will evaluate an already generated pkl file for filtering for a particular scenario `scenario_configs_eval6_v1/jhu_dump/JHU_poisoning_v0_audio_p10_undefended_pytorch_dump.json`."
echo "If you wish to re-compute the file or run for another scenario, please run steps described in README.md"
armory run $extra_args scenario_configs_eval6_v1/jhu_filtering/JHUM_poisoning_v0_audio_p10_dino_clustering_filter.json

