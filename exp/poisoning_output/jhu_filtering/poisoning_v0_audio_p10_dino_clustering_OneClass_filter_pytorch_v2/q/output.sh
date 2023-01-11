#!/bin/bash
cd /Volumes/Macintosh_HD-Data/Documents/GARD/january-2023-submission
. ./path.sh
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  echo -n '# '; cat <<EOF
utils/armory_clsp_poisoning.sh --ncpu 1 --ngpu 1 --armory-opts "" exp/poisoning_output/jhu_filtering/poisoning_v0_audio_p10_dino_clustering_OneClass_filter_pytorch_v2/config.json 
EOF
) >exp/poisoning_output/jhu_filtering/poisoning_v0_audio_p10_dino_clustering_OneClass_filter_pytorch_v2/log/output.log
time1=`date +"%s"`
 ( utils/armory_clsp_poisoning.sh --ncpu 1 --ngpu 1 --armory-opts "" exp/poisoning_output/jhu_filtering/poisoning_v0_audio_p10_dino_clustering_OneClass_filter_pytorch_v2/config.json  ) 2>>exp/poisoning_output/jhu_filtering/poisoning_v0_audio_p10_dino_clustering_OneClass_filter_pytorch_v2/log/output.log >>exp/poisoning_output/jhu_filtering/poisoning_v0_audio_p10_dino_clustering_OneClass_filter_pytorch_v2/log/output.log
ret=$?
time2=`date +"%s"`
echo '#' Accounting: time=$(($time2-$time1)) threads=1 >>exp/poisoning_output/jhu_filtering/poisoning_v0_audio_p10_dino_clustering_OneClass_filter_pytorch_v2/log/output.log
echo '#' Finished at `date` with status $ret >>exp/poisoning_output/jhu_filtering/poisoning_v0_audio_p10_dino_clustering_OneClass_filter_pytorch_v2/log/output.log
[ $ret -eq 137 ] && exit 100;
touch exp/poisoning_output/jhu_filtering/poisoning_v0_audio_p10_dino_clustering_OneClass_filter_pytorch_v2/q/sync/done.27946
exit $[$ret ? 1 : 0]
## submitted with:
# qsub -v PATH -cwd -S /bin/bash -j y -l arch=*64* -o exp/poisoning_output/jhu_filtering/poisoning_v0_audio_p10_dino_clustering_OneClass_filter_pytorch_v2/q/output.log -l hostname="b1[123456789]|c[01]*" -l gpu=1 -l mem_free=16G,ram_free=16G     /Volumes/Macintosh_HD-Data/Documents/GARD/january-2023-submission/exp/poisoning_output/jhu_filtering/poisoning_v0_audio_p10_dino_clustering_OneClass_filter_pytorch_v2/q/output.sh >>exp/poisoning_output/jhu_filtering/poisoning_v0_audio_p10_dino_clustering_OneClass_filter_pytorch_v2/q/output.log 2>&1
