model=ast
model_type=base
dataset=speechcommands

imagenet_pretrain=False

ast_pretrain=True
ast_pretrain_path=/mnt/lynx2/users/mhamza/audiomamba/exp/AST-B_audioset/models/best_audio_model.pth # Modify according to yours
ast_fstride=16
ast_tstride=16
ast_label_dim=527
load_backbone_only=True

bal=none
lr=2.5e-4

epoch=30
freqm=48
timem=48
mixup=0.6
batch_size=128
fstride=16
tstride=16

dataset_mean=-6.845978
dataset_std=5.5654526
audio_length=128
noise=True

metrics=acc
loss=BCE
warmup=False
lrscheduler_start=5
lrscheduler_step=1
lrscheduler_decay=0.85

n_class=35

tr_data=./data/datafiles/speechcommand_train_data.json
val_data=./data/datafiles/speechcommand_valid_data.json
eval_data=./data/datafiles/speechcommand_eval_data.json
label_csv=./data/speechcommands_class_labels_indices.csv

exp_root=/mnt/lynx2/users/mhamza/audiomamba/exp # modify according to yours
exp_name=ast-base_audioset-spc_v2

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
--data-train ${tr_data} --data-val ${val_data} --data-eval ${eval_data} --exp-dir $exp_dir \
--label-csv ${label_csv} --n_class ${n_class} \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --imagenet_pretrain ${imagenet_pretrain} \
--metrics ${metrics} --loss ${loss} --warmup ${warmup} --lrscheduler_start ${lrscheduler_start} --lrscheduler_step ${lrscheduler_step} --lrscheduler_decay ${lrscheduler_decay} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --audio_length ${audio_length} --noise ${noise} \
--ast_pretrain ${ast_pretrain} --ast_pretrain_path ${ast_pretrain_path} --ast_fstride ${ast_fstride} --ast_tstride ${ast_tstride} --ast_label_dim ${ast_label_dim} --load_backbone_only ${load_backbone_only} \
--exp-name ${exp_name} --model_type ${model_type}
