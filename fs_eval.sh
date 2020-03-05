export PYTHONPATH=./:$PYTHONPATH
for dataset in cifar10 svhn cifar100
    do
        for method in natural madry bat feature_scatter gce
            do
                wait
                model_dir=./ckpts/${method}_${dataset}/
                CUDA_VISIBLE_DEVICES=0 nohup python fs_eval.py \
                    --model_dir=$model_dir \
                    --init_model_pass=latest \
                    --attack=True \
                    --attack_method_list=natural-fgsm-pgd20-pgd100-cw20-cw100 \
                    --dataset=${dataset} \
                    --batch_size_test=80 \
                    --resume > log_eval_${method}_${dataset}_old.txt &

                CUDA_VISIBLE_DEVICES=1 nohup python feature_attack_batch.py \
                    --model_dir=$model_dir \
                    --init_model_pass=latest \
                    --dataset=${dataset} \
                    --batch_size_test=100 \
                    --num_attack_step=20 \
                    --resume > log_eval_${method}_${dataset}_new20.txt &

                CUDA_VISIBLE_DEVICES=2 nohup python feature_attack_batch.py \
                    --model_dir=$model_dir \
                    --init_model_pass=latest \
                    --dataset=${dataset} \
                    --batch_size_test=100 \
                    --num_attack_step=100 \
                    --resume > log_eval_${method}_${dataset}_new100.txt &
            done
    done