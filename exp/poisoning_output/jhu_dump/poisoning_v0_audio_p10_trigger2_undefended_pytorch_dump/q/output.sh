#!/bin/bash
cd /Volumes/Macintosh_HD-Data/Documents/GARD/january-2023-submission
. ./path.sh
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  echo -n '# '; cat <<EOF
utils/armory_clsp_poisoning_dump.sh --ncpu 1 --ngpu 0 --armory-opts "" exp/poisoning_output/jhu_dump/poisoning_v0_audio_p10_trigger2_undefended_pytorch_dump/config.json 
EOF
) >exp/poisoning_output/jhu_dump/poisoning_v0_audio_p10_trigger2_undefended_pytorch_dump/log/output.log
time1=`date +"%s"`
 ( utils/armory_clsp_poisoning_dump.sh --ncpu 1 --ngpu 0 --armory-opts "" exp/poisoning_output/jhu_dump/poisoning_v0_audio_p10_trigger2_undefended_pytorch_dump/config.json  ) 2>>exp/poisoning_output/jhu_dump/poisoning_v0_audio_p10_trigger2_undefended_pytorch_dump/log/output.log >>exp/poisoning_output/jhu_dump/poisoning_v0_audio_p10_trigger2_undefended_pytorch_dump/log/output.log
ret=$?
time2=`date +"%s"`
echo '#' Accounting: time=$(($time2-$time1)) threads=1 >>exp/poisoning_output/jhu_dump/poisoning_v0_audio_p10_trigger2_undefended_pytorch_dump/log/output.log
echo '#' Finished at `date` with status $ret >>exp/poisoning_output/jhu_dump/poisoning_v0_audio_p10_trigger2_undefended_pytorch_dump/log/output.log
[ $ret -eq 137 ] && exit 100;
touch exp/poisoning_output/jhu_dump/poisoning_v0_audio_p10_trigger2_undefended_pytorch_dump/q/sync/done.29699
exit $[$ret ? 1 : 0]
## submitted with:
# qsub -v PATH -cwd -S /bin/bash -j y -l arch=*64* -o exp/poisoning_output/jhu_dump/poisoning_v0_audio_p10_trigger2_undefended_pytorch_dump/q/output.log -l hostname="[bc][01]*"  -l mem_free=8G,ram_free=8G    /Volumes/Macintosh_HD-Data/Documents/GARD/january-2023-submission/exp/poisoning_output/jhu_dump/poisoning_v0_audio_p10_trigger2_undefended_pytorch_dump/q/output.sh >>exp/poisoning_output/jhu_dump/poisoning_v0_audio_p10_trigger2_undefended_pytorch_dump/q/output.log 2>&1
