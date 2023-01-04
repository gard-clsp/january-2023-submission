#!/bin/bash
# Copyright
#                2020   Johns Hopkins University (Author: Jesus Villalba, Sonal Joshi)
# Apache 2.0.
#
. ./cmd.sh
set -e

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

if [ $ngpu -eq 0 ]; then
    cmd="$cpu_cmd --num-threads $ncpu"
else
    cmd="$cuda_cmd -l gpu=$ngpu --num-threads $ncpu"
fi

# echo "GPU reserve command is..."
# echo $cmd

# Undefended poisoning attacked baseline : fraction_poisoned=0.1
if [ $stage -le 1 ] && [ $stop_stage -gt 1 ]; then
  exp_dir=exp/poisoning_output/jhu_dump
  label0=${cfg_label}_audio_p10_trigger2_undefended_pytorch_dump
  label=${label0}
  output_dir=$exp_dir/$label
  mkdir -p $output_dir/log
  cp $scenario_config_dir/$label0.json $output_dir/config.json
  echo "running exp $label"
    (
      $cmd $output_dir/log/output.log \
           utils/armory_clsp_poisoning_dump.sh --ncpu $ncpu --ngpu $ngpu \
           --armory-opts "$armory_opts" \
           $output_dir/config.json
      local/retrieve_result.sh $output_dir
    ) &
fi