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

echo "run eval"
armory run $extra_args scenario_configs_eval6_v1/jhu_filtering/poisoning_v0_audio_p10_dino_clustering_filter.json

