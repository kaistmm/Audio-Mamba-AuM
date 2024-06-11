model=ast
model_type=base
dataset=audioset

bal=none
lr=5e-5
epoch=25
tr_data=./data/datafiles/balanced.json
lrscheduler_start=10
lrscheduler_step=5
lrscheduler_decay=0.5

te_data=./data/datafiles/eval.json
label_csv=./data/class_labels_indices.csv
freqm=48
timem=192
mixup=0.5
# corresponding to overlap of 6 for 16*16 patches
fstride=16
tstride=16
batch_size=12

dataset_mean=-4.2677393
dataset_std=4.5689974
audio_length=1024
noise=False

n_class=527

metrics=mAP
loss=BCE
warmup=True

exp_root=/mnt/lynx2/users/mhamza/audiomamba/exp # Modify according to yours
exp_name=ast-base_scratch-as_20k

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
--tstride $tstride --fstride $fstride \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --audio_length ${audio_length} --noise ${noise} \
--metrics ${metrics} --loss ${loss} --warmup ${warmup} --lrscheduler_start ${lrscheduler_start} --lrscheduler_step ${lrscheduler_step} --lrscheduler_decay ${lrscheduler_decay} \
--exp-name ${exp_name} --model_type ${model_type}