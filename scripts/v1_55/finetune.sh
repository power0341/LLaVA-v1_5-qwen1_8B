#!/bin/bash

OMP_NUM_THREADS=1 deepspeed --include=localhost:4,5,6,7 --master_port 23002 llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /data1/xly/models/TinyLlama-1.1B-Chat-v1.0 \
    --version llava_llama_2 \
    --data_path /hotdata/xly/llava-data/LLaVA-Instruct-150K/llava_v1_5_mix665k.json \
    --image_folder /hotdata/xly/llava-data/LLaVA-Instruct-150K/images \
    --vision_tower /data1/xly/models/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter /data1/xly/models/llava-v1.55/llava-mlp2x-336px-tinyllama-1.1b-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /data1/xly/models/llava-v1.55/llava-mlp2x-336px-tinyllama-1.1b-finetune \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard