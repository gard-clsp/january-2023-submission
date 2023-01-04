#!/bin/bash
# Copyright
#                2020   Johns Hopkins University (Author: Jesus Villalba, Sonal Joshi)
# Apache 2.0.
#
. ./cmd.sh
set -e

# [WARNING!]: Change these to excecute only particular systems; otherwise it will run all systems
stage=4
stop_stage=5

# Set-up paths and labels
cfg_label=poisoning_v0
scenario_config_dir=official_scenario_configs_eval6
baseline_defense_subdir=baseline_defense

ncpu=1
ngpu=1
armory_opts=""

. utils/parse_options.sh || exit 1

if [ $ngpu -eq 0 ]; then
    cmd="$cpu_cmd --num-threads $ncpu"
else
    cmd="$cuda_cmd -l gpu=$ngpu --num-threads $ncpu"
fi
echo "GPU reserve command is..."
echo $cmd

# Undefended unattacked baseline : fraction_poisoned=00
if [ $stage -le 1 ] && [ $stop_stage -gt 1 ]; then
  exp_dir=exp/poisoning_output/baseline
  label0=${cfg_label}_audio_p00_undefended
  label=${label0}
  output_dir=$exp_dir/$label
  mkdir -p $output_dir/log
  cp $scenario_config_dir/$label0.json $output_dir/config.json
  echo "running exp $label"
    (
      $cmd $output_dir/log/output.log \
           utils/armory_clsp_poisoning.sh --ncpu $ncpu --ngpu $ngpu \
	   --armory-opts "$armory_opts" \
           $output_dir/config.json
      local/retrieve_result.sh $output_dir
    ) &
fi


# Undefended poisoning attacked baseline : fraction_poisoned=0.01
if [ $stage -le 2 ] && [ $stop_stage -gt 2 ]; then
  exp_dir=exp/poisoning_output/baseline
  label0=${cfg_label}_audio_p01_undefended
  label=${label0}
  output_dir=$exp_dir/$label
  mkdir -p $output_dir/log
  cp $scenario_config_dir/$label0.json $output_dir/config.json
  echo "running exp $label"
    (
      $cmd $output_dir/log/output.log \
           utils/armory_clsp_poisoning.sh --ncpu $ncpu --ngpu $ngpu \
           --armory-opts "$armory_opts" \
           $output_dir/config.json
      local/retrieve_result.sh $output_dir
    ) &
fi


# Undefended poisoning attacked baseline : fraction_poisoned=0.05
if [ $stage -le 3 ] && [ $stop_stage -gt 3 ]; then
  exp_dir=exp/poisoning_output/baseline
  label0=${cfg_label}_audio_p05_undefended.json
  label=${label0}
  output_dir=$exp_dir/$label
  mkdir -p $output_dir/log
  cp $scenario_config_dir/$label0.json $output_dir/config.json
  echo "running exp $label"
    (
      $cmd $output_dir/log/output.log \
           utils/armory_clsp_poisoning.sh --ncpu $ncpu --ngpu $ngpu \
           --armory-opts "$armory_opts" \
           $output_dir/config.json
      local/retrieve_result.sh $output_dir
    ) &
fi


# Undefended poisoning attacked baseline : fraction_poisoned=0.1
if [ $stage -le 4 ] && [ $stop_stage -gt 4 ]; then
  exp_dir=exp/poisoning_output/baseline
  #label0=${cfg_label}_audio_p10_undefended
  label0=${cfg_label}_audio_p10_undefended_v2
  label=${label0}
  output_dir=$exp_dir/$label
  mkdir -p $output_dir/log
  cp $scenario_config_dir/$label0.json $output_dir/config.json
  echo "running exp $label"
    (
      $cmd $output_dir/log/output.log \
           utils/armory_clsp_poisoning.sh --ncpu $ncpu --ngpu $ngpu \
           --armory-opts "$armory_opts" \
           $output_dir/config.json
      local/retrieve_result.sh $output_dir
    ) &
fi

# Undefended poisoning attacked baseline : fraction_poisoned=0.2
if [ $stage -le 5 ] && [ $stop_stage -gt 5 ]; then
  exp_dir=exp/poisoning_output/baseline
  label0=${cfg_label}_audio_p20_undefended
  label=${label0}
  output_dir=$exp_dir/$label
  mkdir -p $output_dir/log
  cp $scenario_config_dir/$label0.json $output_dir/config.json
  echo "running exp $label"
    (
      $cmd $output_dir/log/output.log \
           utils/armory_clsp_poisoning.sh --ncpu $ncpu --ngpu $ngpu \
           --armory-opts "$armory_opts" \
           $output_dir/config.json
      local/retrieve_result.sh $output_dir
    ) &
fi

# Undefended poisoning attacked baseline : fraction_poisoned=0.3
if [ $stage -le 6 ] && [ $stop_stage -gt 6 ]; then
  exp_dir=exp/poisoning_output/baseline
  label0=${cfg_label}_audio_p30_undefended
  label=${label0}
  output_dir=$exp_dir/$label
  mkdir -p $output_dir/log
  cp $scenario_config_dir/$label0.json $output_dir/config.json
  echo "running exp $label"
    (
      $cmd $output_dir/log/output.log \
           utils/armory_clsp_poisoning.sh --ncpu $ncpu --ngpu $ngpu \
           --armory-opts "$armory_opts" \
           $output_dir/config.json
      local/retrieve_result.sh $output_dir
    ) &
fi

# Baseline defense 1 (Activation defense) for poisoning attack with fraction_poisoned=0.1
if [ $stage -le 7 ] && [ $stop_stage -gt 7 ]; then
  exp_dir=exp/poisoning_output/baseline_defended
  label0=${cfg_label}_audio_p10_activation_defense
  label=${label0}
  output_dir=$exp_dir/$label
  mkdir -p $output_dir/log
  cp $scenario_config_dir/$baseline_defense_subdir/$label0.json $output_dir/config.json
  echo "running exp $label"
    (
      $cmd $output_dir/log/output.log \
           utils/armory_clsp_poisoning.sh --ncpu $ncpu --ngpu $ngpu \
           --armory-opts "$armory_opts" \
           $output_dir/config.json
      local/retrieve_result.sh $output_dir
    ) &
fi

# Baseline defense 2 (Perfect Filter) for poisoning attack with fraction_poisoned=0.1
if [ $stage -le 8 ] && [ $stop_stage -gt 8 ]; then
  exp_dir=exp/poisoning_output/baseline_defended
  label0=${cfg_label}_audio_p10_perfect_filter
  label=${label0}
  output_dir=$exp_dir/$label
  mkdir -p $output_dir/log
  cp $scenario_config_dir/$baseline_defense_subdir/$label0.json $output_dir/config.json
  echo "running exp $label"
    (
      $cmd $output_dir/log/output.log \
           utils/armory_clsp_poisoning.sh --ncpu $ncpu --ngpu $ngpu \
           --armory-opts "$armory_opts" \
           $output_dir/config.json
      local/retrieve_result.sh $output_dir
    ) &
fi

# Baseline defense 3 (Random Filter) for poisoning attack with fraction_poisoned=0.1
if [ $stage -le 9 ] && [ $stop_stage -gt 9 ]; then
  exp_dir=exp/poisoning_output/baseline_defended
  label0=${cfg_label}_audio_p10_random_filter
  label=${label0}
  output_dir=$exp_dir/$label
  mkdir -p $output_dir/log
  cp $scenario_config_dir/$baseline_defense_subdir/$label0.json $output_dir/config.json
  echo "running exp $label"
    (
      $cmd $output_dir/log/output.log \
           utils/armory_clsp_poisoning.sh --ncpu $ncpu --ngpu $ngpu \
           --armory-opts "$armory_opts" \
           $output_dir/config.json
      local/retrieve_result.sh $output_dir
    ) &
fi

# Baseline defense 4 (Spectral Signature) for poisoning attack with fraction_poisoned=0.1
if [ $stage -le 10 ] && [ $stop_stage -gt 10 ]; then
  exp_dir=exp/poisoning_output/baseline_defended
  label0=${cfg_label}_audio_p10_spectral_signature_defense
  label=${label0}
  output_dir=$exp_dir/$label
  mkdir -p $output_dir/log
  cp $scenario_config_dir/$baseline_defense_subdir/$label0.json $output_dir/config.json
  echo "running exp $label"
    (
      $cmd $output_dir/log/output.log \
           utils/armory_clsp_poisoning.sh --ncpu $ncpu --ngpu $ngpu \
           --armory-opts "$armory_opts" \
           $output_dir/config.json
      local/retrieve_result.sh $output_dir
    ) &
fi
