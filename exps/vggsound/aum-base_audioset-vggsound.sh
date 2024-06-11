model=aum
model_type=base
dataset=vggsound

aum_pretrain=True
aum_pretrain_path=/mnt/lynx2/users/mhamza/audiomamba/exp/aum-B_audioset/models/best_audio_model.pth # Modify according to yours
aum_pretrain_fstride=16
aum_pretrain_tstride=16

bal=full
lr=1e-5
epoch=20
tr_data=./data/datafiles/vgg_train.json
lrscheduler_start=5
lrscheduler_step=2
lrscheduler_decay=0.75

te_data=./data/datafiles/vgg_test.json
freqm=48
timem=192
mixup=0

fstride=16
tstride=16
batch_size=12

label_csv=./data/class_labels_indices.csv

dataset_mean=-5.0767093
dataset_std=4.4533687
audio_length=1024
noise=False

metrics=acc
loss=BCE
warmup=True
bimamba_type=v1

skip_norm=False
n_class=309

exp_root=/mnt/lynx2/users/mhamza/audiomamba # modify according to yours
exp_name=aum-base_audioset-vggsound

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

CUDA_VISIBLE_DEVICES=0 CUDA_CACHE_DISABLE=1 accelerate launch --mixed_precision=fp16 ../../src/run.py --model ${model} --dataset ${dataset} \
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
--label-csv ${label_csv} --n_class ${n_class} \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --aum_pretrain $aum_pretrain --aum_pretrain_path $aum_pretrain_path --aum_pretrain_fstride $aum_pretrain_fstride --aum_pretrain_tstride $aum_pretrain_tstride \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --audio_length ${audio_length} --noise ${noise} \
--metrics ${metrics} --loss ${loss} --warmup ${warmup} --lrscheduler_start ${lrscheduler_start} --lrscheduler_step ${lrscheduler_step} --lrscheduler_decay ${lrscheduler_decay} \
--exp-name ${exp_name} --model_type ${model_type} --bimamba_type ${bimamba_type}