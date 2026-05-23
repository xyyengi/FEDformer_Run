#!/bin/bash
# FEDformer 服务器部署脚本
# 小波变换版本，周尺度预测（168小时）

# 设置GPU
export CUDA_VISIBLE_DEVICES=0

# 运行训练和预测
python run.py \
    --is_training 1 \
    --task_id wind_solar_load \
    --model FEDformer \
    --version Wavelets \
    --mode_select random \
    --modes 128 \
    --data custom \
    --root_path ./ \
    --data_path Wind_Solar_Load_Processed.csv \
    --features M \
    --target Load \
    --freq h \
    --seq_len 336 \
    --label_len 168 \
    --pred_len 168 \
    --enc_in 11 \
    --dec_in 11 \
    --c_out 11 \
    --d_model 512 \
    --n_heads 8 \
    --e_layers 2 \
    --d_layers 1 \
    --d_ff 2048 \
    --moving_avg 24 168 \
    --factor 1 \
    --distil \
    --dropout 0.05 \
    --embed timeF \
    --activation gelu \
    --loss huber \
    --train_epochs 20 \
    --patience 5 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --num_workers 4 \
    --itr 1 \
    --des Exp_Server_Wavelets \
    --full_inference \
    --use_gpu

echo "训练完成！"
echo "结果保存在: results/wind_solar_load_*_Exp_Server_Wavelets_0/"
echo "实验日志保存在: experiments/"