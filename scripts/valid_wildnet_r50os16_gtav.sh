#!/usr/bin/env bash
echo "Running inference"
domains='vineyard_real'

for target in $domains; do # Iterate on domains
    for i in 1; do # Multiple runs
        date
        echo "Training: i=$i, target=$target"
        python valid.py \
                --dataset agriseg \
                --val_dataset agriseg \
                --wild_dataset imagenet \
                --arch network.lraspp.lraspp_mobilenet_v3_large \
                --city_mode 'train' \
                --sgd \
                --lr_schedule poly \
                --lr 0.00005 \
                --poly_exp 0.9 \
                --max_cu_epoch 10000 \
                --class_uniform_pct 0.5 \
                --class_uniform_tile 1024 \
                --crop_size 224 \
                --scale_min 0.5 \
                --scale_max 2.0 \
                --rrotate 0 \
                --max_iter 15000 \
                --bs_mult 16 \
                --gblur \
                --color_aug 0.5 \
                --fs_layer 1 1 1 0 0 \
                --cont_proj_head 256 \
                --wild_cont_dict_size 393216 \
                --lambda_cel 0.1 \
                --lambda_sel 1.0 \
                --lambda_scr 10.0 \
                --date 1707 \
                --exp lraspp_agriseg_wildnet \
                --ckpt ./logs/ \
                --tb_path ./logs/ \
                --target $target \
                --num_workers 24 \
                --val_perc 1.0 \
                --iou_threshold 0.5
    done
done