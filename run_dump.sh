#!/bin/bash
# Copyright
#                2020   Johns Hopkins University (Author: Jesus Villalba, Sonal Joshi)
# Apache 2.0.
#

# [WARNING!]: Change these to excecute only particular systems; otherwise it will run all systems
stage=1
stop_stage=2

# Set-up paths and labels
cfg_label=poisoning_v0
scenario_config_dir=scenario_configs_eval6_v1/jhu_dump
baseline_defense_subdir=baseline_defense

ncpu=1
ngpu=0
armory_opts=""

. utils/parse_options.sh || exit 1

if [ $ngpu -eq 0 ];then
    export CUDA_VISIBLE_DEVICES=""
    extra_args="--no-gpu"
else
    export CUDA_VISIBLE_DEVICES=1
fi

# Undefended poisoning attacked baseline : fraction_poisoned=0.1
if [ $stage -le 1 ] && [ $stop_stage -gt 1 ]; then
  exp_dir=exp/poisoning_output/jhu_dump
  label0=${cfg_label}_audio_p10_trigger2_undefended_pytorch_dump
  label=${label0}
  output_dir=$exp_dir/$label
  mkdir -p $output_dir/log
  cp $scenario_config_dir/$label0.json $output_dir/config.json
  armory run --check $extra_args $output_dir/config.json

fi