model=aum
model_type=base
dataset=speechcommands

aum_pretrain=True
aum_pretrain_path=/mnt/lynx2/users/mhamza/audiomamba/exp/Aum-B_spc_v2/models/best_audio_model.pth # modify according to yours
aum_pretrain_fstride=16
aum_pretrain_tstride=16

run_type=eval
aum_type=Fo-Bi

batch_size=128
fstride=16
tstride=16

dataset_mean=-6.845978
dataset_std=5.5654526
audio_length=128

metrics=acc
loss=BCE

n_class=35

eval_data=./data/datafiles/speechcommand_eval_data.json
label_csv=./data/speechcommands_class_labels_indices.csv

exp_root=/mnt/lynx2/users/mhamza/audiomamba/exp # modify according to yours
exp_name=aum-base-eval-spc_v2

exp_dir=$exp_root/$exp_name

if [ -d $exp_dir ]; then
  echo "The experiment directory exists. Should I remove it? [y/k/n]"
  read answer
  if [ $answer == "y" ]; then
    rm -r $exp_dir
  elif [ $answer == "k" ]; then
    echo "Keeping the directory"
  else
    echo "Please remove the directory or change the name in the script and run again"
    exit 1
  fi
fi

mkdir -p $exp_dir

python ./prep_sc.py

CUDA_VISIBLE_DEVICES=0 CUDA_CACHE_DISABLE=1 accelerate launch --mixed_precision=fp16 ../../src/run.py --model ${model} --dataset ${dataset} \
--data-val ${eval_data} --exp-dir $exp_dir \
--label-csv ${label_csv} --n_class ${n_class} \
--batch-size $batch_size --save_model True \
--tstride $tstride --fstride $fstride --aum_pretrain $aum_pretrain --aum_pretrain_path $aum_pretrain_path --aum_pretrain_fstride $aum_pretrain_fstride --aum_pretrain_tstride $aum_pretrain_tstride \
--metrics ${metrics} --loss ${loss} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --audio_length ${audio_length} \
--exp-name ${exp_name} --model_type ${model_type} --run_type ${run_type} --aum_type ${aum_type}
