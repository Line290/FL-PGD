#!/usr/bin/env bash
export PYTHONPATH=./:$PYTHONPATH
for dataset in cifar10 svhn cifar100
    do
        for method in natural madry bat feature_scatter gce
            do
                model_dir=./ckpts/${method}_${dataset}/
                mkdir -p $model_dir
                if [ ${dataset} != svhn ]
                then
                    lr=0.1
                else
                    lr=0.01
                fi
                CUDA_VISIBLE_DEVICES=0 python fs_main.py \
                    --resume \
                    --adv_mode=${method} \
                    --lr=0.1 \
                    --model_dir=$model_dir \
                    --init_model_pass=latest \
                    --max_epoch=200 \
                    --save_epochs=100 \
                    --decay_epoch1=60 \
                    --decay_epoch2=90 \
                    --batch_size_train=60 \
                    --dataset=${dataset}
            done
    done
