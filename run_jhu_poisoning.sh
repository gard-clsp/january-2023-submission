#!/bin/bash
# Copyright
#                2020   Johns Hopkins University (Author: Jesus Villalba, Sonal Joshi)
# Apache 2.0.
#
. ./cmd.sh
set -e

# [WARNING!]: Change these to excecute only particular systems; otherwise it will run all systems
stage=7
stop_stage=100

# Set-up paths and labels
cfg_label=poisoning_v0
scenario_config_dir=scenario_configs_eval6_v1

ncpu=1
ngpu=1
armory_opts=""

. utils/parse_options.sh || exit 1

if [ $ngpu -eq 0 ]; then
    cmd="$cpu_cmd --num-threads $ncpu"
else
    cmd="$cuda_cmd -l gpu=$ngpu --num-threads $ncpu"
fi

# echo "GPU reserve command is..."
# echo $cmd

# JHU ResNet50 baseline : No poisoning
if [ $stage -le 1 ] && [ $stop_stage -gt 1 ]; then
  exp_dir=exp/poisoning_output/jhu_baseline
  label0=${cfg_label}_audio_p00_undefended_pytorch
  label=${label0}
  output_dir=$exp_dir/$label
  mkdir -p $output_dir/log
  cp $scenario_config_dir/jhu_baseline/$label0.json $output_dir/config.json
  echo "running exp $label"
    (
      $cmd $output_dir/log/output.log \
           utils/armory_clsp_poisoning.sh --ncpu $ncpu --ngpu $ngpu \
           --armory-opts "$armory_opts" \
           $output_dir/config.json
      local/retrieve_result.sh $output_dir
    ) &
fi

# JHU ResNet50 baseline : Poisoning 10% of class 11 poisoned and added to class 2
if [ $stage -le 2 ] && [ $stop_stage -gt 2 ]; then
  exp_dir=exp/poisoning_output/jhu_baseline
  label0=${cfg_label}_audio_p10_undefended_pytorch
  label=${label0}
  output_dir=$exp_dir/$label
  mkdir -p $output_dir/log
  cp $scenario_config_dir/jhu_baseline/$label0.json $output_dir/config.json
  echo "running exp $label"
    (
      $cmd $output_dir/log/output.log \
           utils/armory_clsp_poisoning.sh --ncpu $ncpu --ngpu $ngpu \
           --armory-opts "$armory_opts" \
           $output_dir/config.json
      local/retrieve_result.sh $output_dir
    ) &
fi

# JHU ResNet50 baseline : Poisoning 10% of class 11 poisoned and added to class 2
# with Perfect Filter filtering defence
if [ $stage -le 3 ] && [ $stop_stage -gt 3 ]; then
  exp_dir=exp/poisoning_output/jhu_baseline
  label0=${cfg_label}_audio_p10_perfect_filter_pytorch
  label=${label0}
  output_dir=$exp_dir/$label
  mkdir -p $output_dir/log
  cp $scenario_config_dir/jhu_baseline/$label0.json $output_dir/config.json
  echo "running exp $label"
    (
      $cmd $output_dir/log/output.log \
           utils/armory_clsp_poisoning.sh --ncpu $ncpu --ngpu $ngpu \
           --armory-opts "$armory_opts" \
           $output_dir/config.json
      local/retrieve_result.sh $output_dir
    ) &
fi

# JHU ResNet50 baseline : Poisoning 10% of class 11 poisoned and added to class 2
# with Random Filter filtering defence
if [ $stage -le 4 ] && [ $stop_stage -gt 4 ]; then
  exp_dir=exp/poisoning_output/jhu_baseline
  label0=${cfg_label}_audio_p10_random_filter_pytorch
  label=${label0}
  output_dir=$exp_dir/$label
  mkdir -p $output_dir/log
  cp $scenario_config_dir/jhu_baseline/$label0.json $output_dir/config.json
  echo "running exp $label"
    (
      $cmd $output_dir/log/output.log \
           utils/armory_clsp_poisoning.sh --ncpu $ncpu --ngpu $ngpu \
           --armory-opts "$armory_opts" \
           $output_dir/config.json
      local/retrieve_result.sh $output_dir
    ) &
fi

# JHU ResNet50 baseline : Poisoning 10% of class 11 poisoned and added to class 2
# with  Spectral Signature defence
if [ $stage -le 5 ] && [ $stop_stage -gt 5 ]; then
  exp_dir=exp/poisoning_output/jhu_baseline
  label0=${cfg_label}_audio_p10_spectral_signature_defense_pytorch
  label=${label0}
  output_dir=$exp_dir/$label
  mkdir -p $output_dir/log
  cp $scenario_config_dir/jhu_baseline/$label0.json $output_dir/config.json
  echo "running exp $label"
    (
      $cmd $output_dir/log/output.log \
           utils/armory_clsp_poisoning.sh --ncpu $ncpu --ngpu $ngpu \
           --armory-opts "$armory_opts" \
           $output_dir/config.json
      local/retrieve_result.sh $output_dir
    ) &
fi

# JHU ResNet50 baseline : Poisoning 10% of class 11 poisoned and added to class 2
# with  Activation defence
if [ $stage -le 6 ] && [ $stop_stage -gt 6 ]; then
  exp_dir=exp/poisoning_output/jhu_baseline
  label0=${cfg_label}_audio_p10_activation_defense_pytorch
  label=${label0}
  output_dir=$exp_dir/$label
  mkdir -p $output_dir/log
  cp $scenario_config_dir/jhu_baseline/$label0.json $output_dir/config.json
  echo "running exp $label"
    (
      $cmd $output_dir/log/output.log \
           utils/armory_clsp_poisoning.sh --ncpu $ncpu --ngpu $ngpu \
           --armory-opts "$armory_opts" \
           $output_dir/config.json
      local/retrieve_result.sh $output_dir
    ) &
fi


# JHU ResNet50 baseline with DINO Clustering filtering defence
if [ $stage -le 7 ] && [ $stop_stage -gt 7 ]; then
  exp_dir=exp/poisoning_output/jhu_filtering
  label0=${cfg_label}_audio_p10_dino_clustering_filter_pytorch
  label=${label0}
  output_dir=$exp_dir/$label
  mkdir -p $output_dir/log
  cp $scenario_config_dir/jhu_filtering/$label0.json $output_dir/config.json
  echo "running exp $label"
    (
      $cmd $output_dir/log/output.log \
           utils/armory_clsp_poisoning.sh --ncpu $ncpu --ngpu $ngpu \
           --armory-opts "$armory_opts" \
           $output_dir/config.json
      local/retrieve_result.sh $output_dir
    ) &
fi

exit

# Defense using DINO_Clustering with fraction_poisoned=0.1; with tensorflow baseline
if [ $stage -le 5 ] && [ $stop_stage -gt 5 ]; then
  exp_dir=exp/poisoning_output/jhu_defences
  defense_subdir=jhu_filtering
  label0=${cfg_label}_audio_p10_dino_clustering_filter
  label=${label0}
  output_dir=$exp_dir/$label
  mkdir -p $output_dir/log
  cp $scenario_config_dir/$defense_subdir/$label0.json $output_dir/config.json
  echo "running exp $label"
    (
      $cmd $output_dir/log/output.log \
           utils/armory_clsp_poisoning.sh --ncpu $ncpu --ngpu $ngpu \
           --armory-opts "$armory_opts" \
           $output_dir/config.json
      local/retrieve_result.sh $output_dir
    ) &
fi
