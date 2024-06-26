model=ast
model_type=base
dataset=epic_sounds

imagenet_pretrain=False

lr=0.00001
epoch=30

freqm=48
timem=192
# corresponding to overlap of 6 for 16*16 patches
fstride=16
tstride=16
batch_size=12

audio_length=1024

metrics=acc
loss=CE
warmup=True

n_class=44

exp_root=/mnt/lynx2/users/mhamza/audiomamba/exp # Modify according to yours
exp_name=ast-base_scratch-epic_sounds

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
--exp-dir $exp_dir --n_class ${n_class} \
--lr $lr --n-epochs ${epoch} --batch-size ${batch_size} --save_model True \
--tstride $tstride --fstride $fstride --imagenet_pretrain $imagenet_pretrain \
--audio_length ${audio_length} --metrics ${metrics} --loss ${loss} --warmup ${warmup} \
--freqm ${freqm} --timem ${timem} \
--exp-name ${exp_name} --model_type ${model_type}